"""Ablation experiments on the fairness weight."""

import pandas as pd

from ..greedy.fairness_aware import fairness_aware_greedy
from ..metrics import evaluate_link_subset_monthly, min_k_for_requirement
from ..model import LinkGraph

def coarse_ablation(graph: LinkGraph, lambdas: list[float]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for lam in lambdas:
        hist = fairness_aware_greedy(graph, lam=lam)
        hist["lambda"] = lam
        frames.append(hist)
    return pd.concat(frames, ignore_index=True)

def fine_ablation(graph: LinkGraph, lambdas: list[float]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for lam in lambdas:
        hist = fairness_aware_greedy(graph, lam=lam)
        hist["lambda"] = lam
        hist["sensor_gap_pct"] = (
            hist["median_sensor_coverage_pct"] - hist["worst_sensor_coverage_pct"]
        )
        frames.append(hist)
    return pd.concat(frames, ignore_index=True)

def summarize_at_ref_k(
    fine_df: pd.DataFrame, lambdas: list[float], ref_ks: list[int]
) -> pd.DataFrame:
    rows: list[dict] = []
    for lam in lambdas:
        for k in ref_ks:
            row = fine_df[(fine_df["lambda"] == lam) & (fine_df["k_links"] == k)]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append(
                {
                    "lambda": lam,
                    "k_links": k,
                    "packet_coverage_pct": r["packet_coverage_pct"],
                    "worst_sensor_coverage_pct": r["worst_sensor_coverage_pct"],
                    "median_sensor_coverage_pct": r["median_sensor_coverage_pct"],
                    "sensor_gap_pct": r["sensor_gap_pct"],
                }
            )
    return pd.DataFrame(rows)

def check_subset_identity(
    fine_df: pd.DataFrame, lambdas: list[float], ref_ks: list[int]
) -> pd.DataFrame:
    rows: list[dict] = []
    for k in ref_ks:
        subsets: dict[float, frozenset] = {}
        for lam in lambdas:
            prefix = fine_df[
                (fine_df["lambda"] == lam) & (fine_df["k_links"] <= k)
            ].sort_values("k_links")
            if len(prefix) < k:
                continue
            subsets[lam] = frozenset(prefix["link_added"].tolist())

        if not subsets:
            continue

        lam_ref = max(subsets.keys())
        ref = subsets[lam_ref]
        for lam, subset in subsets.items():
            rows.append(
                {
                    "k_links": k,
                    "lambda": lam,
                    "subset_size": len(subset),
                    "matches_reference_lambda": lam_ref,
                    "subset_equals_reference": subset == ref,
                    "symmetric_diff_vs_reference": len(subset ^ ref),
                }
            )
    return pd.DataFrame(rows)

def build_sensor_gap_table(
    fine_df: pd.DataFrame, lambdas: list[float], ref_ks: list[int]
) -> pd.DataFrame:
    rows: list[dict] = []
    for lam in lambdas:
        sub = fine_df[fine_df["lambda"] == lam]
        for k in ref_ks:
            row = sub[sub["k_links"] == k]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append(
                {
                    "lambda": lam,
                    "k_links": k,
                    "median_sensor_coverage_pct": r["median_sensor_coverage_pct"],
                    "worst_sensor_coverage_pct": r["worst_sensor_coverage_pct"],
                    "sensor_gap_pct": r["sensor_gap_pct"],
                }
            )
    return pd.DataFrame(rows)

def monthly_stability_of_chosen_subset(
    graph: LinkGraph,
    fairness_greedy_df: pd.DataFrame,
    requirement: tuple[int, int],
) -> tuple[pd.DataFrame, int]:
    chosen_k = min_k_for_requirement(fairness_greedy_df, *requirement)
    if chosen_k is None:
        chosen_k = graph.n_links
    chosen_links = fairness_greedy_df[fairness_greedy_df["k_links"] <= chosen_k][
        "link_added"
    ].tolist()
    return evaluate_link_subset_monthly(graph, chosen_links), chosen_k
