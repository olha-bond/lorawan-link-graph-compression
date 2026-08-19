"""Checks for the generated dataset provenance manifest."""

from pathlib import Path

import pandas as pd

from lgc.io import build_dataset_manifest


def test_manifest_records_utc_months_and_identity_diagnostic(tmp_path: Path) -> None:
    source = tmp_path / "metadata.parquet"
    source.write_bytes(b"test")
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-31T23:59:00Z"),
                "sensor": "s1",
                "gateway": "A",
                "counter": 1,
                "month": "2024-01",
                "link_id": "s1→A",
            },
            {
                "timestamp": pd.Timestamp("2024-01-31T23:59:02Z"),
                "sensor": "s1",
                "gateway": "B",
                "counter": 1,
                "month": "2024-01",
                "link_id": "s1→B",
            },
        ]
    )

    manifest = build_dataset_manifest(df, source, suspicious_span_seconds=1)

    assert manifest["time_handling"]["calendar_month_timezone"] == "UTC"
    assert manifest["dimensions"]["n_unique_packets_by_sensor_counter"] == 1
    assert manifest["packet_identity"]["n_pairs_spanning_multiple_months"] == 0
    assert manifest["packet_identity"]["n_pairs_with_span_above_threshold"] == 1
