"""Coverage measures and link-subset evaluators."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .model import LinkGraph

def evaluate_link_subset(
    graph: LinkGraph, selected_links: Iterable[str], name: str = "subset"
) -> dict:
    selected = list(selected_links)
    d = graph.df[graph.df["link_id"].isin(selected)]
    sub_packets = (
        d.groupby(["sensor", "counter"])
        .agg(best_rssi=("rssi", "max"), best_snr=("snr", "max"))
        .reset_index()
    )

    sensor_totals = pd.DataFrame(
        {
            "sensor": list(graph.sensor_total.keys()),
            "n_unique_packets": list(graph.sensor_total.values()),
        }
    )

    per_sensor = (
        sub_packets.groupby("sensor")
        .size()
        .rename("n_covered")
        .reset_index()
        .merge(sensor_totals, on="sensor", how="right")
    )
    per_sensor["n_covered"] = per_sensor["n_covered"].fillna(0)
    per_sensor["coverage_pct"] = (
        100 * per_sensor["n_covered"] / per_sensor["n_unique_packets"]
    )

    return {
        "selection_name": name,
        "k_links": len(selected),
        "packet_coverage_pct": 100 * len(sub_packets) / graph.total_unique_packets,
        "worst_sensor_coverage_pct": per_sensor["coverage_pct"].min(),
        "median_sensor_coverage_pct": per_sensor["coverage_pct"].median(),
        "mean_sensor_coverage_pct": per_sensor["coverage_pct"].mean(),
        "best_rssi_median": sub_packets["best_rssi"].median()
        if len(sub_packets)
        else np.nan,
        "best_snr_median": sub_packets["best_snr"].median()
        if len(sub_packets)
        else np.nan,
    }

def evaluate_link_subset_monthly(
    graph: LinkGraph, selected_links: Iterable[str]
) -> pd.DataFrame:
    selected = list(selected_links)
    d = graph.df[graph.df["link_id"].isin(selected)]
    sub_packets = (
        d.groupby(["sensor", "counter"]).agg(month=("month", "min")).reset_index()
    )

    monthly_total = graph.all_packets.groupby("month").size().rename("n_total_month")
    monthly_covered = sub_packets.groupby("month").size().rename("n_covered_month")
    monthly = pd.concat([monthly_total, monthly_covered], axis=1).fillna(0)
    monthly["packet_coverage_pct"] = (
        100 * monthly["n_covered_month"] / monthly["n_total_month"]
    )

    sensor_month_totals = (
        graph.all_packets.groupby(["sensor", "month"])
        .size()
        .rename("n_packets_month")
        .reset_index()
    )
    sensor_month_covered = (
        sub_packets.groupby(["sensor", "month"])
        .size()
        .rename("n_covered")
        .reset_index()
    )
    merged = sensor_month_totals.merge(
        sensor_month_covered, on=["sensor", "month"], how="left"
    )
    merged["n_covered"] = merged["n_covered"].fillna(0)
    merged["coverage_pct"] = 100 * merged["n_covered"] / merged["n_packets_month"]

    worst = (
        merged.groupby("month")["coverage_pct"]
        .min()
        .rename("worst_sensor_coverage_pct")
    )
    median = (
        merged.groupby("month")["coverage_pct"]
        .median()
        .rename("median_sensor_coverage_pct")
    )
    return monthly.join(worst).join(median).reset_index().sort_values("month")

def min_k_for_requirement(
    per_k_df: pd.DataFrame,
    P_min: float,
    S_min: float,
    k_col: str = "k_links",
    cov_col: str = "packet_coverage_pct",
    worst_col: str = "worst_sensor_coverage_pct",
) -> int | None:
    ok = per_k_df[(per_k_df[cov_col] >= P_min) & (per_k_df[worst_col] >= S_min)]
    return int(ok[k_col].min()) if not ok.empty else None

def row_at_k(ordering_df: pd.DataFrame, k: int, k_col: str = "k_links") -> pd.Series:
    return ordering_df[ordering_df[k_col] == k].iloc[0]

def selected_up_to(
    ordering_df: pd.DataFrame,
    k: int,
    k_col: str = "k_links",
    link_col: str = "link_added",
) -> list[str]:
    return sorted(ordering_df[ordering_df[k_col] <= k][link_col].tolist())
