"""Tests for the manuscript-facing publication bundle exporter."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.export_publication_bundle import (
    S1_COLUMNS,
    S2_COLUMNS,
    _write_supplementary_tables,
)


def _rolling_row() -> dict[str, object]:
    values: dict[str, object] = {
        "test_month": "2025-01",
        "P_min_pct": 90,
        "S_min_pct": 80,
        "policy": "expanding_window",
        "fit_method": "proposed",
        "fit_months": "2024-02;2024-03",
        "fit_status": "Feasible",
        "n_links": 10,
        "packet_coverage_pct": 95.0,
        "worst_sensor_coverage_pct": 85.0,
        "median_sensor_coverage_pct": 96.0,
        "n_active_sensors": 9,
        "meets_packet_threshold": True,
        "meets_sensor_threshold": True,
        "meets_both_thresholds": True,
        "test_oracle_n_links": 9,
        "budget_minus_test_oracle": 1,
        "subset_jaccard_with_test_oracle": 0.8,
        "subset_hash": "abc123",
        "selected_links": "sensor01→gatewayA",
        "is_initial_observation_cold_start": False,
        "is_before_frozen_policy_available": False,
    }
    assert set(S1_COLUMNS).union(S2_COLUMNS).issubset(values)
    return values


def test_supplementary_tables_are_created(tmp_path: Path) -> None:
    source = tmp_path / "outputs"
    destination = tmp_path / "bundle"
    source.mkdir()
    destination.mkdir()
    pd.DataFrame([_rolling_row()]).to_csv(
        source / "30_rolling_temporal_per_month.csv", index=False
    )

    _write_supplementary_tables(source, destination)

    s1 = pd.read_csv(
        destination / "supplementary" / "Table_S1_per_month_feasibility_and_link_budget.csv"
    )
    s2 = pd.read_csv(
        destination / "supplementary" / "Table_S2_oracle_budget_and_subset_similarity.csv"
    )
    assert list(s1.columns) == S1_COLUMNS
    assert list(s2.columns) == S2_COLUMNS


def test_supplementary_export_rejects_missing_columns(tmp_path: Path) -> None:
    source = tmp_path / "outputs"
    destination = tmp_path / "bundle"
    source.mkdir()
    destination.mkdir()
    pd.DataFrame([{"test_month": "2025-01"}]).to_csv(
        source / "30_rolling_temporal_per_month.csv", index=False
    )

    with pytest.raises(ValueError, match="missing required columns"):
        _write_supplementary_tables(source, destination)
