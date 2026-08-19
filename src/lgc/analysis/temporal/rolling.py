"""Prospective rolling policies and their summary statistics."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...model import LinkGraph
from .context import combine_window
from .fitting import evaluate_subset_on_month, fit_from_combined_window, subset_hash
from .models import CombinedWindow, FitResult, TemporalContext

_PROSPECTIVE_METHODS = ("proposed", "multicover", "coverage_only", "exact")
_FROZEN_METHODS = ("proposed", "multicover", "coverage_only")


@dataclass
class PolicyFit:
    policy: str
    method: str
    fit_months: list[str]
    result: FitResult


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else float("nan")


def _rolling_row(
    graph: LinkGraph,
    context: TemporalContext,
    policy_fit: PolicyFit,
    test_month: str,
    test_index: int,
    P_min: float,
    S_min: float,
    oracle: FitResult,
    frozen_initial_months: int,
) -> dict:
    result = policy_fit.result
    evaluation = evaluate_subset_on_month(
        graph,
        context,
        result.selected_links,
        test_month,
    )
    meets_packet = evaluation.packet_coverage_pct >= P_min
    meets_sensor = evaluation.worst_sensor_coverage_pct >= S_min

    oracle_links = set(oracle.selected_links)
    selected_links = set(result.selected_links)
    budget_difference = (
        result.n_links - oracle.n_links
        if result.n_links is not None and oracle.n_links is not None
        else float("nan")
    )
    if meets_packet and meets_sensor and oracle.n_links is not None:
        assert result.n_links is not None
        assert result.n_links >= oracle.n_links

    return {
        "test_month": test_month,
        "P_min_pct": P_min,
        "S_min_pct": S_min,
        "policy": policy_fit.policy,
        "fit_method": policy_fit.method,
        "fit_months": ";".join(policy_fit.fit_months),
        "fit_status": result.status,
        "n_links": result.n_links,
        "selected_links": ";".join(result.selected_links),
        "subset_hash": subset_hash(result.selected_links) if result.selected_links else "",
        **evaluation.to_record(),
        "meets_packet_threshold": bool(meets_packet),
        "meets_sensor_threshold": bool(meets_sensor),
        "meets_both_thresholds": bool(meets_packet and meets_sensor),
        "test_oracle_n_links": oracle.n_links,
        "budget_minus_test_oracle": budget_difference,
        "subset_jaccard_with_test_oracle": _jaccard(
            selected_links,
            oracle_links,
        ),
        "is_initial_observation_cold_start": test_index == 1,
        "is_before_frozen_policy_available": test_index < frozen_initial_months,
    }


def run_rolling_prospective(
    graph: LinkGraph,
    context: TemporalContext,
    requirements: Iterable[tuple[int, int]],
    lam: float = 1.0,
    frozen_initial_months: int = 3,
) -> pd.DataFrame:
    if len(context.months) < 2:
        return pd.DataFrame()
    if not 1 <= frozen_initial_months < len(context.months):
        raise ValueError("frozen_initial_months must be in [1, n_months-1].")

    requirements = list(requirements)
    frozen_months = context.months[:frozen_initial_months]
    frozen_window = combine_window(graph, context, frozen_months)
    frozen_results = {
        (P_min, S_min, method): fit_from_combined_window(
            graph,
            frozen_window,
            P_min,
            S_min,
            method=method,
            lam=lam,
        )
        for P_min, S_min in requirements
        for method in _FROZEN_METHODS
    }

    rows: list[dict] = []
    for test_index, test_month in enumerate(context.months[1:], start=1):
        previous_months = [context.months[test_index - 1]]
        expanding_months = context.months[:test_index]
        previous_window = combine_window(graph, context, previous_months)
        expanding_window = combine_window(graph, context, expanding_months)
        oracle_window = combine_window(graph, context, [test_month])

        for P_min, S_min in requirements:
            oracle = fit_from_combined_window(
                graph,
                oracle_window,
                P_min,
                S_min,
                method="exact",
                lam=lam,
            )
            policy_fits: list[PolicyFit] = []
            for method in _PROSPECTIVE_METHODS:
                policy_fits.append(
                    PolicyFit(
                        policy="previous_month",
                        method=method,
                        fit_months=previous_months,
                        result=fit_from_combined_window(
                            graph,
                            previous_window,
                            P_min,
                            S_min,
                            method=method,
                            lam=lam,
                        ),
                    )
                )
                policy_fits.append(
                    PolicyFit(
                        policy="expanding_window",
                        method=method,
                        fit_months=expanding_months,
                        result=fit_from_combined_window(
                            graph,
                            expanding_window,
                            P_min,
                            S_min,
                            method=method,
                            lam=lam,
                        ),
                    )
                )

            if test_index >= frozen_initial_months:
                policy_fits.extend(
                    PolicyFit(
                        policy="frozen_initial_window",
                        method=method,
                        fit_months=frozen_months,
                        result=frozen_results[(P_min, S_min, method)],
                    )
                    for method in _FROZEN_METHODS
                )

            policy_fits.append(
                PolicyFit(
                    policy="retrospective_oracle",
                    method="exact",
                    fit_months=[test_month],
                    result=oracle,
                )
            )

            rows.extend(
                _rolling_row(
                    graph,
                    context,
                    policy_fit,
                    test_month,
                    test_index,
                    P_min,
                    S_min,
                    oracle,
                    frozen_initial_months,
                )
                for policy_fit in policy_fits
            )

    return pd.DataFrame(rows)


def _new_sensor_months(context: TemporalContext) -> set[str]:
    first_month = context.months[0]
    first_active: dict[str, str] = {}
    for month in context.months:
        for sensor, count in context.sensor_total_by_month[month].items():
            if count > 0:
                first_active.setdefault(sensor, month)
    return {month for month in first_active.values() if month != first_month}


def _evaluation_windows(
    context: TemporalContext,
    frozen_initial_months: int,
) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    first_test = context.months[1]
    common_start = context.months[frozen_initial_months]
    cold_start_months = _new_sensor_months(context)

    # windows differ by policy availability
    return {
        "all_available_policy_months": lambda frame: frame,
        "common_after_initial_window": lambda frame: frame[
            frame["test_month"] >= common_start
        ],
        "common_excluding_new_sensor_cold_starts": lambda frame: frame[
            (frame["test_month"] >= common_start)
            & (~frame["test_month"].isin(cold_start_months))
        ],
        "excluding_initial_observation_cold_start": lambda frame: frame[
            frame["test_month"] != first_test
        ],
    }


def _parse_links(value: str) -> set[str]:
    return set(value.split(";")) if value else set()


def _transition_statistics(feasible: pd.DataFrame) -> dict[str, float | int]:
    jaccards: list[float] = []
    symmetric_differences: list[int] = []
    previous: set[str] | None = None

    for value in feasible.sort_values("test_month")["selected_links"].fillna(""):
        current = _parse_links(value)
        if previous is not None:
            union = current | previous
            jaccards.append(len(current & previous) / len(union) if union else 1.0)
            symmetric_differences.append(len(current ^ previous))
        previous = current

    return {
        "n_reconfigurations": int(
            sum(difference > 0 for difference in symmetric_differences)
        ),
        "mean_consecutive_subset_jaccard": (
            float(np.mean(jaccards)) if jaccards else float("nan")
        ),
        "mean_links_changed_per_reoptimization": (
            float(np.mean(symmetric_differences))
            if symmetric_differences
            else float("nan")
        ),
        "max_links_changed_in_one_reoptimization": (
            int(max(symmetric_differences)) if symmetric_differences else 0
        ),
    }


def _mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if len(frame) else float("nan")


def _min_or_nan(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].min()) if len(frame) else float("nan")


def _max_or_nan(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].max()) if len(frame) else float("nan")


def _summary_row(
    keys: tuple,
    window_name: str,
    subset: pd.DataFrame,
) -> dict:
    P_min, S_min, policy, method = keys
    feasible = subset[subset["n_links"].notna()].copy()
    passing = feasible[feasible["meets_both_thresholds"]].copy()

    return {
        "P_min_pct": int(P_min),
        "S_min_pct": int(S_min),
        "policy": policy,
        "fit_method": method,
        "evaluation_window": window_name,
        "n_test_months": int(len(subset)),
        "n_infeasible_fits": int(len(subset) - len(feasible)),
        "n_meets_both": int(subset["meets_both_thresholds"].fillna(False).sum()),
        "pass_rate_both": float(
            subset["meets_both_thresholds"].fillna(False).mean()
        ),
        "pass_rate_packet": float(
            subset["meets_packet_threshold"].fillna(False).mean()
        ),
        "pass_rate_sensor": float(
            subset["meets_sensor_threshold"].fillna(False).mean()
        ),
        "mean_n_links": _mean_or_nan(feasible, "n_links"),
        "mean_test_packet_coverage_pct": _mean_or_nan(
            feasible,
            "packet_coverage_pct",
        ),
        "min_test_packet_coverage_pct": _min_or_nan(
            feasible,
            "packet_coverage_pct",
        ),
        "mean_test_worst_sensor_coverage_pct": _mean_or_nan(
            feasible,
            "worst_sensor_coverage_pct",
        ),
        "min_test_worst_sensor_coverage_pct": _min_or_nan(
            feasible,
            "worst_sensor_coverage_pct",
        ),
        "mean_extra_budget_over_oracle_when_passing": _mean_or_nan(
            passing,
            "budget_minus_test_oracle",
        ),
        "max_extra_budget_over_oracle_when_passing": _max_or_nan(
            passing,
            "budget_minus_test_oracle",
        ),
        "mean_subset_jaccard_with_test_oracle": _mean_or_nan(
            feasible,
            "subset_jaccard_with_test_oracle",
        ),
        **_transition_statistics(feasible),
    }


def summarize_rolling(
    rolling: pd.DataFrame,
    context: TemporalContext,
    frozen_initial_months: int = 3,
) -> pd.DataFrame:
    if rolling.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    windows = _evaluation_windows(context, frozen_initial_months)
    group_columns = ["P_min_pct", "S_min_pct", "policy", "fit_method"]

    for keys, group in rolling.groupby(group_columns, sort=True):
        for window_name, select_window in windows.items():
            subset = select_window(group).copy()
            if not subset.empty:
                rows.append(_summary_row(keys, window_name, subset))

    return pd.DataFrame(rows)
