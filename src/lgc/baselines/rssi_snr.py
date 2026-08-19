"""Top-*k* RSSI/SNR baseline."""

import pandas as pd

from ..io import minmax
from ..metrics import evaluate_link_subset
from ..model import LinkGraph

def build_rssi_snr_ranking(link_summary: pd.DataFrame) -> pd.DataFrame:
    df = link_summary.copy()
    df["rssi_mean_norm"] = minmax(df["rssi_mean"])
    df["snr_mean_norm"] = minmax(df["snr_mean"])
    df["signal_quality_score"] = 0.5 * df["rssi_mean_norm"] + 0.5 * df["snr_mean_norm"]
    if "link_id" not in df.columns:
        df["link_id"] = df["sensor"] + "→" + df["gateway"]
    return df.sort_values(
        ["signal_quality_score", "link_id"],
        ascending=[False, True],
        kind="mergesort",
    )

def top_k_rssi_snr_dense(graph: LinkGraph, link_summary: pd.DataFrame) -> pd.DataFrame:
    ranked = build_rssi_snr_ranking(link_summary)["link_id"].tolist()
    rows = [
        evaluate_link_subset(graph, ranked[:k], f"top{k}_rssi_snr")
        for k in range(1, len(graph.all_link_ids) + 1)
    ]
    return pd.DataFrame(rows)
