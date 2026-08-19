"""Per-link statistics across months."""

import pandas as pd

def build_monthly_link_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month", "sensor", "gateway"])
        .agg(
            n_records=("rssi", "size"),
            rssi_mean=("rssi", "mean"),
            rssi_std=("rssi", "std"),
            snr_mean=("snr", "mean"),
            snr_std=("snr", "std"),
            sf_mean=("sf", "mean"),
            airtime_mean=("airtime", "mean"),
        )
        .reset_index()
    )

def build_link_temporal_stability(monthly_link: pd.DataFrame) -> pd.DataFrame:
    stability = (
        monthly_link.groupby(["sensor", "gateway"])
        .agg(
            active_months=("month", "nunique"),
            monthly_n_records_mean=("n_records", "mean"),
            monthly_n_records_std=("n_records", "std"),
            monthly_rssi_mean=("rssi_mean", "mean"),
            monthly_rssi_std_across_months=("rssi_mean", "std"),
            monthly_snr_mean=("snr_mean", "mean"),
            monthly_snr_std_across_months=("snr_mean", "std"),
            monthly_sf_mean=("sf_mean", "mean"),
            monthly_airtime_mean=("airtime_mean", "mean"),
        )
        .reset_index()
    )
    stability["rssi_temporal_cv"] = (
        stability["monthly_rssi_std_across_months"]
        / stability["monthly_rssi_mean"].abs()
    )
    stability["snr_temporal_cv"] = (
        stability["monthly_snr_std_across_months"] / stability["monthly_snr_mean"].abs()
    )
    return stability.sort_values(
        ["active_months", "monthly_rssi_mean"], ascending=[False, False]
    )
