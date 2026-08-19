import pandas as pd

from lgc.baselines.reliability import (
    build_reliability_ranking,
    reliability_ranking_from_utility,
)


def test_reliability_ranking_contains_link_id():
    link_summary = pd.DataFrame({
        "sensor": ["s1", "s1"],
        "gateway": ["g1", "g2"],
        "rssi_mean": [-90.0, -80.0],
        "snr_mean": [2.0, 6.0],
    })
    sensor_gateway_coverage = pd.DataFrame({
        "sensor": ["s1", "s1"],
        "gateway": ["g1", "g2"],
        "coverage_pct_of_sensor_packets": [70.0, 90.0],
    })
    link_stability = pd.DataFrame({
        "sensor": ["s1", "s1"],
        "gateway": ["g1", "g2"],
        "active_months": [12, 12],
        "monthly_rssi_std_across_months": [2.0, 1.0],
        "monthly_snr_std_across_months": [1.0, 0.5],
        "rssi_temporal_cv": [0.04, 0.02],
        "snr_temporal_cv": [0.20, 0.10],
    })

    ranking = build_reliability_ranking(link_summary, sensor_gateway_coverage, link_stability)
    assert ranking["link_id"].tolist() == ["s1→g2", "s1→g1"]
    assert reliability_ranking_from_utility(ranking) == ["s1→g2", "s1→g1"]
