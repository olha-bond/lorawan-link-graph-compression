"""Enumeration of gateway subsets and their coverage."""

import itertools

import numpy as np
import pandas as pd

def evaluate_gateway_subsets(df: pd.DataFrame, packet_rx: pd.DataFrame) -> pd.DataFrame:
    all_gateways = sorted(df["gateway"].dropna().unique())
    sensor_total_packets = (
        packet_rx.groupby("sensor").size().rename("n_unique_packets").reset_index()
    )
    total_unique_packets = len(packet_rx)

    rows: list[dict] = []
    for r in range(1, len(all_gateways) + 1):
        for subset in itertools.combinations(all_gateways, r):
            subset_name = "+".join(subset)
            df_sub = df[df["gateway"].isin(subset)]
            sub_packets = (
                df_sub.groupby(["sensor", "counter"])
                .agg(
                    n_gateways_in_subset=("gateway", "nunique"),
                    best_rssi=("rssi", "max"),
                    best_snr=("snr", "max"),
                    mean_rssi=("rssi", "mean"),
                    mean_snr=("snr", "mean"),
                    min_sf=("sf", "min"),
                    max_sf=("sf", "max"),
                )
                .reset_index()
            )
            per_sensor = (
                sub_packets.groupby("sensor")
                .size()
                .rename("n_packets_covered")
                .reset_index()
                .merge(sensor_total_packets, on="sensor", how="right")
            )
            per_sensor["n_packets_covered"] = per_sensor["n_packets_covered"].fillna(0)
            per_sensor["coverage_pct"] = (
                100 * per_sensor["n_packets_covered"] / per_sensor["n_unique_packets"]
            )

            n_covered = len(sub_packets)
            rows.append(
                {
                    "gateway_subset": subset_name,
                    "n_gateways": len(subset),
                    "n_packets_covered": n_covered,
                    "packet_coverage_pct": 100 * n_covered / total_unique_packets,
                    "sensors_with_any_coverage": (
                        per_sensor["n_packets_covered"] > 0
                    ).sum(),
                    "sensors_with_75pct_coverage": (
                        per_sensor["coverage_pct"] >= 75
                    ).sum(),
                    "sensors_with_90pct_coverage": (
                        per_sensor["coverage_pct"] >= 90
                    ).sum(),
                    "sensors_with_95pct_coverage": (
                        per_sensor["coverage_pct"] >= 95
                    ).sum(),
                    "worst_sensor_coverage_pct": per_sensor["coverage_pct"].min(),
                    "median_sensor_coverage_pct": per_sensor["coverage_pct"].median(),
                    "mean_sensor_coverage_pct": per_sensor["coverage_pct"].mean(),
                    "best_rssi_mean": sub_packets["best_rssi"].mean(),
                    "best_rssi_median": sub_packets["best_rssi"].median(),
                    "best_snr_mean": sub_packets["best_snr"].mean(),
                    "best_snr_median": sub_packets["best_snr"].median(),
                    "mean_redundancy_within_subset": sub_packets[
                        "n_gateways_in_subset"
                    ].mean(),
                    "pct_packets_redundant_within_subset": (
                        100 * (sub_packets["n_gateways_in_subset"] >= 2).mean()
                        if len(sub_packets) > 0
                        else 0
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["n_gateways", "packet_coverage_pct", "worst_sensor_coverage_pct"],
        ascending=[True, False, False],
    )

def evaluate_gateway_subsets_monthly(df: pd.DataFrame) -> pd.DataFrame:
    all_gateways = sorted(df["gateway"].dropna().unique())
    rows: list[dict] = []

    for month, df_month in df.groupby("month"):
        if df_month.empty:
            continue
        packet_month = (
            df_month.groupby(["sensor", "counter"]).size().rename("n").reset_index()
        )
        total_month = len(packet_month)
        if total_month == 0:
            continue
        sensor_total_month = (
            packet_month.groupby("sensor")
            .size()
            .rename("n_unique_packets_month")
            .reset_index()
        )

        for r in range(1, len(all_gateways) + 1):
            for subset in itertools.combinations(all_gateways, r):
                subset_name = "+".join(subset)
                df_sub = df_month[df_month["gateway"].isin(subset)]

                sub_packets = (
                    df_sub.groupby(["sensor", "counter"])
                    .agg(
                        n_gateways_in_subset=("gateway", "nunique"),
                        best_rssi=("rssi", "max"),
                        best_snr=("snr", "max"),
                    )
                    .reset_index()
                )
                per_sensor = (
                    sub_packets.groupby("sensor")
                    .size()
                    .rename("n_packets_covered")
                    .reset_index()
                    .merge(sensor_total_month, on="sensor", how="right")
                )
                per_sensor["n_packets_covered"] = per_sensor[
                    "n_packets_covered"
                ].fillna(0)
                per_sensor["coverage_pct"] = (
                    100
                    * per_sensor["n_packets_covered"]
                    / per_sensor["n_unique_packets_month"]
                )

                rows.append(
                    {
                        "month": month,
                        "gateway_subset": subset_name,
                        "n_gateways": len(subset),
                        "n_packets_covered": len(sub_packets),
                        "packet_coverage_pct": 100 * len(sub_packets) / total_month,
                        "worst_sensor_coverage_pct": per_sensor["coverage_pct"].min(),
                        "median_sensor_coverage_pct": per_sensor[
                            "coverage_pct"
                        ].median(),
                        "mean_sensor_coverage_pct": per_sensor["coverage_pct"].mean(),
                        "best_rssi_median": (
                            sub_packets["best_rssi"].median()
                            if len(sub_packets)
                            else np.nan
                        ),
                        "best_snr_median": (
                            sub_packets["best_snr"].median()
                            if len(sub_packets)
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)

def summarize_monthly_gateway_subsets(monthly_eval: pd.DataFrame) -> pd.DataFrame:
    return (
        monthly_eval.groupby(["gateway_subset", "n_gateways"])
        .agg(
            packet_coverage_mean=("packet_coverage_pct", "mean"),
            packet_coverage_std=("packet_coverage_pct", "std"),
            worst_sensor_coverage_mean=("worst_sensor_coverage_pct", "mean"),
            worst_sensor_coverage_min=("worst_sensor_coverage_pct", "min"),
            median_sensor_coverage_mean=("median_sensor_coverage_pct", "mean"),
            best_rssi_median_mean=("best_rssi_median", "mean"),
            best_snr_median_mean=("best_snr_median", "mean"),
            n_months=("month", "nunique"),
        )
        .reset_index()
        .sort_values(["n_gateways", "packet_coverage_mean"], ascending=[True, False])
    )
