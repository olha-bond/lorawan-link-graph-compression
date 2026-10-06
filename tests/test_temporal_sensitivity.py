import pandas as pd

from lgc.analysis.temporal import (
    build_temporal_context,
    run_boundary_month_sensitivity,
    run_frozen_window_sensitivity,
    subset_temporal_context,
)
from lgc.model import LinkGraph


def _sample_graph(n_months: int = 4) -> LinkGraph:
    rows = []
    months = pd.date_range(
        "2024-01-01",
        periods=n_months,
        freq="MS",
        tz="UTC",
    )
    for month_index, month in enumerate(months):
        month_name = month.strftime("%Y-%m")
        for sensor_index in range(1, 3):
            sensor = f"sensor{sensor_index:02d}"
            for packet_index in range(4):
                counter = month_index * 100 + packet_index
                gateways = ["A"] if packet_index else ["A", "B"]
                for gateway in gateways:
                    rows.append(
                        {
                            "timestamp": month + pd.Timedelta(days=packet_index),
                            "month": month_name,
                            "sensor": sensor,
                            "gateway": gateway,
                            "counter": counter,
                            "rssi": -90.0,
                            "snr": 5.0,
                            "link_id": f"{sensor}→{gateway}",
                        }
                    )
    return LinkGraph(pd.DataFrame(rows))


def test_subset_temporal_context_drops_boundary_months():
    graph = _sample_graph()
    context = build_temporal_context(graph)
    subset = subset_temporal_context(context, context.months[1:-1])

    assert subset.months == ["2024-02", "2024-03"]
    assert set(subset.packet_rows_by_month) == {"2024-02", "2024-03"}
    assert set(subset.sensor_total_by_month) == {"2024-02", "2024-03"}
    assert set(subset.total_packets_by_month) == {"2024-02", "2024-03"}
    assert set(subset.pattern_lookup_by_month) == {"2024-02", "2024-03"}


def test_boundary_month_sensitivity_keeps_robust_budget_on_sample():
    graph = _sample_graph()
    context = build_temporal_context(graph)

    result = run_boundary_month_sensitivity(
        graph,
        context,
        [(90, 80)],
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["all_months_n"] == 4
    assert row["complete_months_n"] == 2
    assert row["excluded_first_month"] == "2024-01"
    assert row["excluded_last_month"] == "2024-04"
    assert row["all_months_status"] == "Optimal"
    assert row["complete_months_status"] == "Optimal"
    assert row["all_months_min_links"] == 2
    assert row["complete_months_min_links"] == 2
    assert row["delta_links"] == 0
    assert row["same_selected_subset"]
    assert row["subset_jaccard"] == 1.0


def test_frozen_window_sensitivity_uses_equal_length_windows():
    graph = _sample_graph(n_months=6)
    context = build_temporal_context(graph)

    summary, per_month = run_frozen_window_sensitivity(
        graph,
        context,
        [(90, 80)],
        frozen_initial_months=3,
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["baseline_fit_months"] == "2024-01;2024-02;2024-03"
    assert row["complete_fit_months"] == "2024-02;2024-03;2024-04"
    assert row["excluded_incomplete_month"] == "2024-01"
    assert row["baseline_test_months_n"] == 3
    assert row["common_test_months_n"] == 2
    assert row["common_test_start"] == "2024-05"
    assert row["common_test_end"] == "2024-06"
    assert len(per_month) == 2
    assert per_month["test_month"].tolist() == ["2024-05", "2024-06"]
