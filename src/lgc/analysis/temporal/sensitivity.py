"""Sensitivity checks for the temporal robustness analysis."""

from collections.abc import Iterable

import pandas as pd

from ...model import LinkGraph
from .fitting import evaluate_subset_on_month, fit_on_window, subset_hash
from .models import TemporalContext
from .robust import solve_exact_all_month_static


def subset_temporal_context(
    context: TemporalContext,
    months: Iterable[str],
) -> TemporalContext:
    month_list = list(months)
    if not month_list:
        raise ValueError("Temporal subset must contain at least one month.")

    unknown = [month for month in month_list if month not in context.months]
    if unknown:
        raise KeyError(f"Unknown month(s): {unknown}")

    return TemporalContext(
        months=month_list,
        packet_rows_by_month={
            month: context.packet_rows_by_month[month] for month in month_list
        },
        sensor_total_by_month={
            month: context.sensor_total_by_month[month] for month in month_list
        },
        total_packets_by_month={
            month: context.total_packets_by_month[month] for month in month_list
        },
        pattern_lookup_by_month={
            month: context.pattern_lookup_by_month[month] for month in month_list
        },
    )


def _subset_jaccard(first: list[str], second: list[str]) -> float:
    first_set = set(first)
    second_set = set(second)
    union = first_set | second_set
    if not union:
        return 1.0
    return len(first_set & second_set) / len(union)


def run_boundary_month_sensitivity(
    graph: LinkGraph,
    context: TemporalContext,
    requirements: Iterable[tuple[int, int]],
) -> pd.DataFrame:
    if len(context.months) < 3:
        raise ValueError("At least three calendar months are required.")

    complete_months = context.months[1:-1]
    complete_context = subset_temporal_context(context, complete_months)
    rows: list[dict] = []

    for P_min, S_min in requirements:
        all_months = solve_exact_all_month_static(
            graph,
            context,
            P_min,
            S_min,
        )
        complete = solve_exact_all_month_static(
            graph,
            complete_context,
            P_min,
            S_min,
        )

        delta_links = None
        if all_months.n_links is not None and complete.n_links is not None:
            delta_links = complete.n_links - all_months.n_links

        rows.append(
            {
                "P_min_pct": P_min,
                "S_min_pct": S_min,
                "all_months_n": len(context.months),
                "complete_months_n": len(complete_months),
                "excluded_first_month": context.months[0],
                "excluded_last_month": context.months[-1],
                "all_months_status": all_months.status,
                "complete_months_status": complete.status,
                "all_months_min_links": all_months.n_links,
                "complete_months_min_links": complete.n_links,
                "delta_links": delta_links,
                "same_selected_subset": (
                    set(all_months.selected_links) == set(complete.selected_links)
                ),
                "subset_jaccard": _subset_jaccard(
                    all_months.selected_links,
                    complete.selected_links,
                ),
                "all_months_selected_links": ";".join(all_months.selected_links),
                "complete_months_selected_links": ";".join(
                    complete.selected_links
                ),
            }
        )

    return pd.DataFrame(rows)


def run_frozen_window_sensitivity(
    graph: LinkGraph,
    context: TemporalContext,
    requirements: Iterable[tuple[int, int]],
    lam: float = 1.0,
    frozen_initial_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 1 <= frozen_initial_months < len(context.months) - 1:
        raise ValueError(
            "frozen_initial_months must leave at least one common test month."
        )

    baseline_months = context.months[:frozen_initial_months]
    complete_months = context.months[1 : frozen_initial_months + 1]
    baseline_test_months = context.months[frozen_initial_months:]
    common_test_months = context.months[frozen_initial_months + 1 :]
    summary_rows: list[dict] = []
    monthly_rows: list[dict] = []

    for P_min, S_min in requirements:
        baseline = fit_on_window(
            graph,
            context,
            baseline_months,
            P_min,
            S_min,
            method="proposed",
            lam=lam,
        )
        complete = fit_on_window(
            graph,
            context,
            complete_months,
            P_min,
            S_min,
            method="proposed",
            lam=lam,
        )

        baseline_all = []
        for month in baseline_test_months:
            evaluation = evaluate_subset_on_month(
                graph,
                context,
                baseline.selected_links,
                month,
            )
            baseline_all.append(
                evaluation.packet_coverage_pct >= P_min
                and evaluation.worst_sensor_coverage_pct >= S_min
            )

        baseline_common = []
        complete_common = []
        for month in common_test_months:
            baseline_evaluation = evaluate_subset_on_month(
                graph,
                context,
                baseline.selected_links,
                month,
            )
            complete_evaluation = evaluate_subset_on_month(
                graph,
                context,
                complete.selected_links,
                month,
            )
            baseline_meets = bool(
                baseline_evaluation.packet_coverage_pct >= P_min
                and baseline_evaluation.worst_sensor_coverage_pct >= S_min
            )
            complete_meets = bool(
                complete_evaluation.packet_coverage_pct >= P_min
                and complete_evaluation.worst_sensor_coverage_pct >= S_min
            )
            baseline_common.append(baseline_meets)
            complete_common.append(complete_meets)

            monthly_rows.append(
                {
                    "test_month": month,
                    "P_min_pct": P_min,
                    "S_min_pct": S_min,
                    "baseline_fit_months": ";".join(baseline_months),
                    "complete_fit_months": ";".join(complete_months),
                    "baseline_n_links": baseline.n_links,
                    "complete_n_links": complete.n_links,
                    "baseline_packet_coverage_pct": (
                        baseline_evaluation.packet_coverage_pct
                    ),
                    "baseline_worst_sensor_coverage_pct": (
                        baseline_evaluation.worst_sensor_coverage_pct
                    ),
                    "baseline_meets_both_thresholds": baseline_meets,
                    "complete_packet_coverage_pct": (
                        complete_evaluation.packet_coverage_pct
                    ),
                    "complete_worst_sensor_coverage_pct": (
                        complete_evaluation.worst_sensor_coverage_pct
                    ),
                    "complete_meets_both_thresholds": complete_meets,
                }
            )

        baseline_rate = sum(baseline_common) / len(baseline_common)
        complete_rate = sum(complete_common) / len(complete_common)
        delta_links = None
        if baseline.n_links is not None and complete.n_links is not None:
            delta_links = complete.n_links - baseline.n_links

        summary_rows.append(
            {
                "P_min_pct": P_min,
                "S_min_pct": S_min,
                "baseline_fit_months": ";".join(baseline_months),
                "complete_fit_months": ";".join(complete_months),
                "excluded_incomplete_month": context.months[0],
                "baseline_status": baseline.status,
                "complete_status": complete.status,
                "baseline_n_links": baseline.n_links,
                "complete_n_links": complete.n_links,
                "delta_links": delta_links,
                "same_selected_subset": (
                    set(baseline.selected_links) == set(complete.selected_links)
                ),
                "subset_jaccard": _subset_jaccard(
                    baseline.selected_links,
                    complete.selected_links,
                ),
                "baseline_subset_hash": (
                    subset_hash(baseline.selected_links)
                    if baseline.selected_links
                    else ""
                ),
                "complete_subset_hash": (
                    subset_hash(complete.selected_links)
                    if complete.selected_links
                    else ""
                ),
                "baseline_selected_links": ";".join(
                    baseline.selected_links
                ),
                "complete_selected_links": ";".join(
                    complete.selected_links
                ),
                "baseline_test_months_n": len(baseline_test_months),
                "baseline_n_meets_both": sum(baseline_all),
                "baseline_pass_rate_both": (
                    sum(baseline_all) / len(baseline_all)
                ),
                "common_test_start": common_test_months[0],
                "common_test_end": common_test_months[-1],
                "common_test_months_n": len(common_test_months),
                "baseline_common_n_meets_both": sum(baseline_common),
                "complete_common_n_meets_both": sum(complete_common),
                "baseline_common_pass_rate_both": baseline_rate,
                "complete_common_pass_rate_both": complete_rate,
                "delta_common_pass_rate_pp": (
                    100.0 * (complete_rate - baseline_rate)
                ),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(monthly_rows)
