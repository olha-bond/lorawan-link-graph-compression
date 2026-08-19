"""All-month static optimization and temporal requirement tables."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from ...metrics import min_k_for_requirement
from ...model import LinkGraph
from ...thresholds import minimum_required_count
from .context import combine_window
from .fitting import evaluate_subset_on_month, fit_from_combined_window
from .models import FitResult, TemporalContext


def _sensor_options_for_all_months(
    graph: LinkGraph,
    context: TemporalContext,
    sensor: str,
    S_min: float,
) -> list[dict]:
    gateways = sorted(graph.sensor_gateway_lists[sensor])
    options = []

    for mask in range(2 ** len(gateways)):
        subset = frozenset(
            gateways[index]
            for index in range(len(gateways))
            if (mask >> index) & 1
        )
        covered_by_month: dict[str, int] = {}
        valid = True

        for month in context.months:
            total = context.sensor_total_by_month[month].get(sensor, 0)
            if total == 0:
                covered_by_month[month] = 0
                continue

            covered = sum(
                count
                for gateway_set, count in context.pattern_lookup_by_month[month][
                    sensor
                ]
                if gateway_set & subset
            )
            if covered < minimum_required_count(total, S_min):
                valid = False
                break
            covered_by_month[month] = int(covered)

        if valid:
            options.append(
                {
                    "cost": len(subset),
                    "links": subset,
                    "covered_by_month": covered_by_month,
                }
            )

    return options


def solve_exact_all_month_static(
    graph: LinkGraph,
    context: TemporalContext,
    P_min: float,
    S_min: float,
) -> FitResult:
    sensor_options = {
        sensor: _sensor_options_for_all_months(graph, context, sensor, S_min)
        for sensor in graph.all_sensors
    }
    if any(not options for options in sensor_options.values()):
        return FitResult("Infeasible", None, [])

    variables: list[tuple[str, int]] = []
    costs: list[float] = []
    for sensor in graph.all_sensors:
        for option_index, option in enumerate(sensor_options[sensor]):
            variables.append((sensor, option_index))
            costs.append(float(option["cost"]))

    n_variables = len(variables)
    sensor_constraints = np.zeros(
        (len(graph.all_sensors), n_variables),
        dtype=float,
    )
    for sensor_row, sensor in enumerate(graph.all_sensors):
        for column, (variable_sensor, _) in enumerate(variables):
            if variable_sensor == sensor:
                sensor_constraints[sensor_row, column] = 1.0

    month_constraints = np.zeros(
        (len(context.months), n_variables),
        dtype=float,
    )
    month_lower_bounds = np.zeros(len(context.months), dtype=float)
    for month_row, month in enumerate(context.months):
        month_lower_bounds[month_row] = minimum_required_count(
            context.total_packets_by_month[month], P_min
        )
        for column, (sensor, option_index) in enumerate(variables):
            option = sensor_options[sensor][option_index]
            month_constraints[month_row, column] = option[
                "covered_by_month"
            ].get(month, 0)

    # One option per sensor, monthly rows enforce aggregate coverage
    result = milp(
        c=np.asarray(costs),
        constraints=[
            LinearConstraint(sensor_constraints, lb=1.0, ub=1.0),
            LinearConstraint(
                month_constraints,
                lb=month_lower_bounds,
                ub=np.inf,
            ),
        ],
        integrality=np.ones(n_variables),
        bounds=Bounds(lb=0.0, ub=1.0),
        options={"mip_rel_gap": 0.0},
    )

    if result.status == 2:
        return FitResult("Infeasible", None, [])
    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"All-month MILP failed with status {result.status}: {result.message}"
        )

    chosen = np.rint(result.x).astype(int)
    selected_links: list[str] = []
    for column, (sensor, option_index) in enumerate(variables):
        if chosen[column] != 1:
            continue
        selected_links.extend(
            f"{sensor}→{gateway}"
            for gateway in sensor_options[sensor][option_index]["links"]
        )
    selected_links.sort()

    for month in context.months:
        evaluation = evaluate_subset_on_month(
            graph,
            context,
            selected_links,
            month,
        )
        if evaluation.packet_coverage_pct < P_min - 1e-7:
            raise RuntimeError("All-month MILP verification failed for packet coverage.")
        if evaluation.worst_sensor_coverage_pct < S_min - 1e-7:
            raise RuntimeError("All-month MILP verification failed for sensor coverage.")

    return FitResult("Optimal", int(round(result.fun)), selected_links)


def build_temporal_requirement_table(
    graph: LinkGraph,
    context: TemporalContext,
    requirements: Iterable[tuple[int, int]],
    retrospective_detail: pd.DataFrame,
    whole_proposed_ordering: pd.DataFrame,
) -> pd.DataFrame:
    whole_window = combine_window(graph, context, context.months)
    rows: list[dict] = []

    for P_min, S_min in requirements:
        whole_exact = fit_from_combined_window(
            graph,
            whole_window,
            P_min,
            S_min,
            method="exact",
        )
        robust = solve_exact_all_month_static(graph, context, P_min, S_min)

        requirement_rows = retrospective_detail[
            (retrospective_detail["P_min_pct"] == P_min)
            & (retrospective_detail["S_min_pct"] == S_min)
        ]
        exact = requirement_rows[
            requirement_rows["method"] == "exact"
        ].set_index("month")
        proposed = requirement_rows[
            requirement_rows["method"] == "proposed"
        ].set_index("month")
        multicover = requirement_rows[
            requirement_rows["method"] == "multicover"
        ].set_index("month")

        common_months = (
            exact.index.intersection(proposed.index).intersection(multicover.index)
        )
        exact_k = exact.loc[common_months, "n_links"]
        proposed_k = proposed.loc[common_months, "n_links"]
        multicover_k = multicover.loc[common_months, "n_links"]

        rows.append(
            {
                "P_min_pct": P_min,
                "S_min_pct": S_min,
                "whole_dataset_proposed_min_links": min_k_for_requirement(
                    whole_proposed_ordering,
                    P_min,
                    S_min,
                ),
                "whole_dataset_exact_min_links": whole_exact.n_links,
                "all_month_static_exact_min_links": robust.n_links,
                "retrospective_monthly_exact_mean_links": float(exact_k.mean()),
                "retrospective_monthly_exact_worst_case_links": float(
                    exact_k.max()
                ),
                "retrospective_monthly_proposed_mean_links": float(
                    proposed_k.mean()
                ),
                "retrospective_monthly_proposed_worst_case_links": float(
                    proposed_k.max()
                ),
                "retrospective_monthly_multicover_mean_links": float(
                    multicover_k.mean()
                ),
                "retrospective_monthly_multicover_worst_case_links": float(
                    multicover_k.max()
                ),
                "proposed_matches_exact_every_month": bool(
                    exact_k.notna().all()
                    and proposed_k.notna().all()
                    and (exact_k.astype(int) == proposed_k.astype(int)).all()
                ),
                "multicover_matches_exact_every_month": bool(
                    exact_k.notna().all()
                    and multicover_k.notna().all()
                    and (exact_k.astype(int) == multicover_k.astype(int)).all()
                ),
            }
        )

    return pd.DataFrame(rows)
