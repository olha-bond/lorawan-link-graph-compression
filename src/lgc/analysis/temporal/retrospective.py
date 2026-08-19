"""Retrospective month-specific link selection and summaries."""

from collections.abc import Iterable

import pandas as pd

from ...model import LinkGraph
from .context import combine_window
from .fitting import evaluate_subset_on_month, fit_from_combined_window, subset_hash
from .models import TemporalContext

_METHODS = ("exact", "proposed", "multicover", "coverage_only")


def run_retrospective_monthly(
    graph: LinkGraph,
    context: TemporalContext,
    requirements: Iterable[tuple[int, int]],
    lam: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict] = []

    for month in context.months:
        window = combine_window(graph, context, [month])
        for P_min, S_min in requirements:
            for method in _METHODS:
                result = fit_from_combined_window(
                    graph,
                    window,
                    P_min,
                    S_min,
                    method=method,
                    lam=lam,
                )
                evaluation = evaluate_subset_on_month(
                    graph,
                    context,
                    result.selected_links,
                    month,
                )
                meets_packet = evaluation.packet_coverage_pct >= P_min
                meets_sensor = evaluation.worst_sensor_coverage_pct >= S_min
                rows.append(
                    {
                        "month": month,
                        "P_min_pct": P_min,
                        "S_min_pct": S_min,
                        "method": method,
                        "fit_status": result.status,
                        "n_links": result.n_links,
                        "selected_links": ";".join(result.selected_links),
                        "subset_hash": (
                            subset_hash(result.selected_links)
                            if result.selected_links
                            else ""
                        ),
                        **evaluation.to_record(),
                        "meets_packet_threshold": bool(meets_packet),
                        "meets_sensor_threshold": bool(meets_sensor),
                        "meets_both_thresholds": bool(meets_packet and meets_sensor),
                    }
                )

    return pd.DataFrame(rows)


def _value_or_nan(series: pd.Series, operation: str) -> float:
    if series.empty:
        return float("nan")
    return float(getattr(series, operation)())


def summarize_retrospective_monthly(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    grouped = detail.groupby(
        ["P_min_pct", "S_min_pct", "method"],
        sort=True,
    )
    for (P_min, S_min, method), group in grouped:
        feasible = group[group["n_links"].notna()]
        rows.append(
            {
                "P_min_pct": int(P_min),
                "S_min_pct": int(S_min),
                "method": method,
                "n_months": int(len(group)),
                "n_feasible_fits": int(len(feasible)),
                "n_months_meeting_both": int(
                    group["meets_both_thresholds"].fillna(False).sum()
                ),
                "pass_rate_both": float(
                    group["meets_both_thresholds"].fillna(False).mean()
                ),
                "mean_n_links": _value_or_nan(feasible["n_links"], "mean"),
                "max_n_links": _value_or_nan(feasible["n_links"], "max"),
                "mean_packet_coverage_pct": _value_or_nan(
                    feasible["packet_coverage_pct"], "mean"
                ),
                "min_packet_coverage_pct": _value_or_nan(
                    feasible["packet_coverage_pct"], "min"
                ),
                "mean_worst_sensor_coverage_pct": _value_or_nan(
                    feasible["worst_sensor_coverage_pct"], "mean"
                ),
                "min_worst_sensor_coverage_pct": _value_or_nan(
                    feasible["worst_sensor_coverage_pct"], "min"
                ),
            }
        )

    return pd.DataFrame(rows)
