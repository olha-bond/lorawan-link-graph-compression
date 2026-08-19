"""Dataset loading, column normalisation, and integrity checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXPECTED_GATEWAYS, EXPECTED_LINKS, EXPECTED_SENSORS

_COLUMN_RENAME = {
    "Timestamp": "timestamp",
    "RSSI (dBm)": "rssi",
    "SNR (dB)": "snr",
    "Spreading Factor (-)": "sf",
    "Bandwidth (Hz)": "bandwidth",
    "Frequency (Hz)": "frequency",
    "Airtime (s)": "airtime",
    "Counter (-)": "counter",
    "# Receiving Gateways (-)": "n_receiving_gateways",
    "Sensor Alias": "sensor",
    "Gateway Alias": "gateway",
}

_REQUIRED_COLUMNS = {"timestamp", "sensor", "gateway", "counter", "rssi", "snr"}
_CRITICAL_COLUMNS = ["timestamp", "sensor", "gateway", "counter"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_COLUMN_RENAME)


def load_dataset(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = normalize_columns(pd.read_parquet(parquet_path))
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {sorted(missing)}"
        )

    nulls = df[_CRITICAL_COLUMNS].isna().sum()
    if (nulls > 0).any():
        raise ValueError(
            "Nulls in critical columns (packet identity would be undefined):\n"
            + nulls[nulls > 0].to_string()
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        n_bad = int(df["timestamp"].isna().sum())
        raise ValueError(
            f"pd.to_datetime failed on {n_bad} timestamps. "
            "Check the raw Timestamp column for malformed values."
        )

    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
    df["hour"] = df["timestamp"].dt.hour
    df["link_id"] = df["sensor"] + "→" + df["gateway"]
    return df


def validate_complete_link_grid(df: pd.DataFrame) -> None:
    sensors = sorted(df["sensor"].dropna().unique())
    gateways = sorted(df["gateway"].dropna().unique())
    observed_links = set(df["link_id"].dropna().unique())
    expected_links = {f"{sensor}→{gateway}" for sensor in sensors for gateway in gateways}
    missing = expected_links - observed_links
    if missing:
        raise RuntimeError(
            f"{len(missing)} sensor--gateway pair(s) are never observed: "
            f"{sorted(missing)[:20]}."
        )


def validate_uva_dataset(df: pd.DataFrame) -> None:
    sensors = sorted(df["sensor"].dropna().unique())
    gateways = sorted(df["gateway"].dropna().unique())
    links = sorted(df["link_id"].dropna().unique())

    if len(sensors) != EXPECTED_SENSORS:
        raise RuntimeError(
            f"Expected {EXPECTED_SENSORS} sensors, dataset contains "
            f"{len(sensors)}: {sensors}."
        )
    if len(gateways) != EXPECTED_GATEWAYS:
        raise RuntimeError(
            f"Expected {EXPECTED_GATEWAYS} gateways, dataset contains "
            f"{len(gateways)}: {gateways}."
        )
    validate_complete_link_grid(df)
    if len(links) != EXPECTED_LINKS:
        raise RuntimeError(
            f"Expected {EXPECTED_LINKS} links, dataset contains {len(links)}."
        )


def check_counter_uniqueness(df: pd.DataFrame) -> None:
    counter_months = df.groupby(["sensor", "counter"])["month"].nunique()
    repeated = counter_months[counter_months > 1]
    if len(repeated) > 0:
        raise RuntimeError(
            f"(sensor, counter) is not globally unique across months: "
            f"{len(repeated)} pairs affected. Redefine packet identity to include "
            "month or a session id, then re-run."
        )



def validate_analysis_dataset(df: pd.DataFrame) -> None:
    """Run all mandatory integrity checks used by publication analyses."""
    validate_uva_dataset(df)
    check_counter_uniqueness(df)


def build_dataset_manifest(
    df: pd.DataFrame,
    source_path: Path,
    suspicious_span_seconds: float = 60.0,
) -> dict:
    """Build a compact, deterministic manifest for the loaded experiment data.

    The same-month span diagnostic does not redefine packet identity. It merely
    records how many ``(sensor, counter)`` groups have receptions spread over a
    long interval, which can reveal possible within-month counter reuse.
    """
    identity_groups = df.groupby(["sensor", "counter"], sort=False).agg(
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        n_months=("month", "nunique"),
    )
    spans = (
        identity_groups["last_timestamp"] - identity_groups["first_timestamp"]
    ).dt.total_seconds()
    months = sorted(df["month"].dropna().astype(str).unique().tolist())
    sensors = sorted(df["sensor"].dropna().astype(str).unique().tolist())
    gateways = sorted(df["gateway"].dropna().astype(str).unique().tolist())
    links = sorted(df["link_id"].dropna().astype(str).unique().tolist())

    source = source_path.expanduser().resolve()
    return {
        "dataset": {
            "name": "UVA long-term LoRaWAN communication metadata",
            "doi": "10.18130/V3/RFTICK",
            "source_file": source.name,
            "expected_relative_path": (
                "dataset/lorawan_metadata/lorawan_combined_dataset.parquet"
            ),
            "source_size_bytes": source.stat().st_size if source.exists() else None,
        },
        "time_handling": {
            "timestamp_timezone": "UTC",
            "calendar_month_timezone": "UTC",
            "first_timestamp_utc": df["timestamp"].min().isoformat(),
            "last_timestamp_utc": df["timestamp"].max().isoformat(),
            "months": months,
            "n_calendar_months": len(months),
        },
        "dimensions": {
            "n_reception_rows": int(len(df)),
            "n_unique_packets_by_sensor_counter": int(len(identity_groups)),
            "n_sensors": len(sensors),
            "sensors": sensors,
            "n_gateways": len(gateways),
            "gateways": gateways,
            "n_observed_links": len(links),
            "observed_links": links,
        },
        "packet_identity": {
            "columns": ["sensor", "counter"],
            "n_pairs_spanning_multiple_months": int(
                (identity_groups["n_months"] > 1).sum()
            ),
            "same_month_counter_reuse_not_excluded": True,
            "suspicious_timestamp_span_seconds": float(suspicious_span_seconds),
            "n_pairs_with_span_above_threshold": int(
                (spans > suspicious_span_seconds).sum()
            ),
            "max_pair_timestamp_span_seconds": float(spans.max())
            if len(spans)
            else 0.0,
        },
    }


def save_json(payload: dict, out_dir: Path, filename: str) -> Path:
    path = out_dir / filename
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {path}")
    return path


def save_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    path = out_dir / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")
    return path


def robust_quantile(values, q: float) -> float:
    series = pd.Series(values).dropna()
    return float(series.quantile(q)) if len(series) else float("nan")


def coefficient_of_variation(values) -> float:
    series = pd.Series(values).dropna()
    if len(series) < 2:
        return float("nan")
    mean = series.mean()
    if mean == 0:
        return float("nan")
    return float(series.std() / abs(mean))


def minmax(values) -> pd.Series:
    series = pd.Series(values, dtype="float64")
    if series.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    minimum, maximum = series.min(), series.max()
    if maximum == minimum:
        return pd.Series(0.5, index=series.index)
    return (series - minimum) / (maximum - minimum)
