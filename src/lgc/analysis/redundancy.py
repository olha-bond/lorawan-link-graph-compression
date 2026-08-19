"""Link, packet, and per-sensor summaries of the reception structure."""

import numpy as np
import pandas as pd

from ..io import robust_quantile

def build_link_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["sensor", "gateway"])
        .agg(
            n_records=("rssi", "size"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
            rssi_mean=("rssi", "mean"),
            rssi_std=("rssi", "std"),
            rssi_q05=("rssi", lambda x: robust_quantile(x, 0.05)),
            rssi_q50=("rssi", lambda x: robust_quantile(x, 0.50)),
            rssi_q95=("rssi", lambda x: robust_quantile(x, 0.95)),
            snr_mean=("snr", "mean"),
            snr_std=("snr", "std"),
            snr_q05=("snr", lambda x: robust_quantile(x, 0.05)),
            snr_q50=("snr", lambda x: robust_quantile(x, 0.50)),
            snr_q95=("snr", lambda x: robust_quantile(x, 0.95)),
            sf_mean=("sf", "mean"),
            sf_mode=("sf", lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan),
            airtime_mean=("airtime", "mean"),
            frequency_nunique=("frequency", "nunique"),
            bandwidth_nunique=("bandwidth", "nunique"),
            counter_min=("counter", "min"),
            counter_max=("counter", "max"),
        )
        .reset_index()
    )
    summary["duration_days"] = (
        summary["last_seen"] - summary["first_seen"]
    ).dt.total_seconds() / 86400
    return summary.sort_values(["sensor", "rssi_mean"], ascending=[True, False])

def build_packet_reception(df: pd.DataFrame) -> pd.DataFrame:
    packet_rx = (
        df.groupby(["sensor", "counter"])
        .agg(
            timestamp=("timestamp", "min"),
            n_gateways=("gateway", "nunique"),
            gateways=("gateway", lambda x: "+".join(sorted(x.dropna().unique()))),
            rssi_best=("rssi", "max"),
            rssi_mean=("rssi", "mean"),
            snr_best=("snr", "max"),
            snr_mean=("snr", "mean"),
            sf_min=("sf", "min"),
            sf_max=("sf", "max"),
            n_rows=("gateway", "size"),
        )
        .reset_index()
    )
    packet_rx["month"] = (
        packet_rx["timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
    )
    return packet_rx

def build_redundancy_distribution(packet_rx: pd.DataFrame) -> pd.DataFrame:
    counts = (
        packet_rx["n_gateways"]
        .value_counts()
        .sort_index()
        .rename_axis("n_gateways")
        .reset_index(name="n_packets")
    )
    counts["pct"] = 100 * counts["n_packets"] / counts["n_packets"].sum()
    return counts

def build_gateway_combination_counts(packet_rx: pd.DataFrame) -> pd.DataFrame:
    combos = (
        packet_rx["gateways"]
        .value_counts()
        .rename_axis("gateway_combo")
        .reset_index(name="n_packets")
    )
    combos["pct"] = 100 * combos["n_packets"] / combos["n_packets"].sum()
    return combos

def build_sensor_gateway_coverage(
    df: pd.DataFrame, packet_rx: pd.DataFrame
) -> pd.DataFrame:
    sensor_total_packets = (
        packet_rx.groupby("sensor").size().rename("n_unique_packets").reset_index()
    )
    sensor_gateway_packets = (
        df.groupby(["sensor", "gateway"])["counter"]
        .nunique()
        .rename("n_packets_received_by_gateway")
        .reset_index()
    )
    cov = sensor_gateway_packets.merge(sensor_total_packets, on="sensor", how="left")
    cov["coverage_pct_of_sensor_packets"] = (
        100 * cov["n_packets_received_by_gateway"] / cov["n_unique_packets"]
    )
    return cov.sort_values(
        ["sensor", "coverage_pct_of_sensor_packets"], ascending=[True, False]
    )
