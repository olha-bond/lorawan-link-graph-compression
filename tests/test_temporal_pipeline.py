from pathlib import Path

import pandas as pd

from scripts import run_temporal_analysis


def _sample_dataset() -> pd.DataFrame:
    rows = []
    months = pd.date_range("2024-01-01", periods=4, freq="MS", tz="UTC")
    for month_index, month in enumerate(months):
        for sensor_index in range(1, 11):
            sensor = f"sensor{sensor_index:02d}"
            for packet_index in range(6):
                counter = month_index * 100 + packet_index
                gateways = ["A"] if packet_index else ["A", "B", "C"]
                for gateway in gateways:
                    rows.append(
                        {
                            "timestamp": month + pd.Timedelta(days=packet_index),
                            "sensor": sensor,
                            "gateway": gateway,
                            "counter": counter,
                            "rssi": -90.0 - sensor_index,
                            "snr": 5.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_temporal_stage_runs_from_loader_to_summary(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    metadata = (
        data_root
        / "dataset"
        / "lorawan_metadata"
        / "lorawan_combined_dataset.parquet"
    )
    metadata.parent.mkdir(parents=True)
    metadata.touch()

    sample = _sample_dataset()
    monkeypatch.setattr("lgc.io.pd.read_parquet", lambda _path: sample.copy())

    return_code = run_temporal_analysis.main(
        [
            "--data-root",
            str(data_root),
            "--out-dir",
            str(output_dir),
            "--frozen-initial-months",
            "2",
        ]
    )

    assert return_code == 0
    expected = [
        "23_retrospective_monthly_reoptimization.csv",
        "23b_retrospective_monthly_reoptimization_summary.csv",
        "24_temporal_requirement_table.csv",
        "24b_per_month_exact_vs_greedy.csv",
        "30_rolling_temporal_per_month.csv",
        "30b_rolling_temporal_summary.csv",
    ]
    assert all((output_dir / name).exists() for name in expected)

    summary = pd.read_csv(output_dir / "30b_rolling_temporal_summary.csv")
    assert not summary.empty
    assert {
        "policy",
        "fit_method",
        "evaluation_window",
        "pass_rate_both",
        "mean_n_links",
        "n_reconfigurations",
    } <= set(summary.columns)
