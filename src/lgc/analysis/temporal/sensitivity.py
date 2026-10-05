"""Sensitivity checks for the temporal robustness analysis."""

from collections.abc import Iterable

import pandas as pd

from ...model import LinkGraph
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
