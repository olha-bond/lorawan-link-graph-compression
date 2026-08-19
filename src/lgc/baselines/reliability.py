"""Top-*k* reliability baseline."""

import pandas as pd

from ..io import minmax
from ..metrics import evaluate_link_subset
from ..model import LinkGraph

RELIABILITY_WEIGHTS = {
    "coverage": 0.30,
    "rssi": 0.20,
    "snr": 0.20,
    "rssi_stability": 0.15,
    "snr_stability": 0.15,
}

def build_reliability_ranking(
    link_summary: pd.DataFrame,
    sensor_gateway_coverage: pd.DataFrame,
    link_stability: pd.DataFrame,
) -> pd.DataFrame:
    utility = link_summary.merge(
        sensor_gateway_coverage[
            ["sensor", "gateway", "coverage_pct_of_sensor_packets"]
        ],
        on=["sensor", "gateway"],
        how="left",
    )

    utility = utility.merge(
        link_stability[
            [
                "sensor",
                "gateway",
                "active_months",
                "monthly_rssi_std_across_months",
                "monthly_snr_std_across_months",
                "rssi_temporal_cv",
                "snr_temporal_cv",
            ]
        ],
        on=["sensor", "gateway"],
        how="left",
    )

    utility["coverage_norm"] = minmax(utility["coverage_pct_of_sensor_packets"])
    utility["rssi_mean_norm"] = minmax(utility["rssi_mean"])
    utility["snr_mean_norm"] = minmax(utility["snr_mean"])
    utility["active_months_norm"] = minmax(utility["active_months"])

    utility["rssi_temporal_stability_norm"] = 1 - minmax(utility["rssi_temporal_cv"])
    utility["snr_temporal_stability_norm"] = 1 - minmax(utility["snr_temporal_cv"])

    utility["exploratory_reliability_score"] = (
        RELIABILITY_WEIGHTS["coverage"] * utility["coverage_norm"]
        + RELIABILITY_WEIGHTS["rssi"] * utility["rssi_mean_norm"]
        + RELIABILITY_WEIGHTS["snr"] * utility["snr_mean_norm"]
        + RELIABILITY_WEIGHTS["rssi_stability"]
        * utility["rssi_temporal_stability_norm"]
        + RELIABILITY_WEIGHTS["snr_stability"] * utility["snr_temporal_stability_norm"]
    )
    utility["link_id"] = utility["sensor"] + "→" + utility["gateway"]

    utility = utility.sort_values(
        ["exploratory_reliability_score", "link_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    return utility

def reliability_ranking_from_utility(utility: pd.DataFrame) -> list[str]:
    df = utility.copy()
    if "link_id" not in df.columns:
        df["link_id"] = df["sensor"] + "→" + df["gateway"]
    return df.sort_values(
        ["exploratory_reliability_score", "link_id"],
        ascending=[False, True],
        kind="mergesort",
    )["link_id"].tolist()

def top_k_reliability_dense(graph: LinkGraph, ranked_links: list[str]) -> pd.DataFrame:
    rows = [
        evaluate_link_subset(graph, ranked_links[:k], f"top{k}_reliability")
        for k in range(1, len(graph.all_link_ids) + 1)
    ]
    return pd.DataFrame(rows)
