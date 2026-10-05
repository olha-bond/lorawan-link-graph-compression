"""Tests for heterogeneous synthetic sensor traffic."""

import numpy as np
import pandas as pd

from lgc.analysis.traffic_heterogeneity import (
    empirical_sensor_traffic_profile,
    generate_synthetic_topology_with_packet_counts,
    synthetic_sensor_packet_counts,
)


def test_empirical_sensor_traffic_profile_uses_unique_observed_packets() -> None:
    df = pd.DataFrame(
        {
            "sensor": ["s1", "s1", "s1", "s2", "s2", "s2", "s2"],
            "gateway": ["g1", "g2", "g1", "g1", "g1", "g1", "g2"],
            "counter": [1, 1, 2, 1, 2, 3, 3],
        }
    )
    out = empirical_sensor_traffic_profile(df)

    assert out.set_index("sensor").loc["s1", "observed_packets"] == 2
    assert out.set_index("sensor").loc["s2", "observed_packets"] == 3
    assert np.isclose(out["traffic_multiplier"].mean(), 1.0)


def test_heterogeneous_packet_counts_preserve_total() -> None:
    multipliers = np.array([0.5, 1.0, 1.5])
    balanced = synthetic_sensor_packet_counts(
        multipliers,
        n_sensors=6,
        packets_per_sensor=100,
        seed=2,
        heterogeneous=False,
    )
    heterogeneous = synthetic_sensor_packet_counts(
        multipliers,
        n_sensors=6,
        packets_per_sensor=100,
        seed=2,
        heterogeneous=True,
    )

    assert balanced.tolist() == [100] * 6
    assert heterogeneous.sum() == 600
    assert heterogeneous.min() < 100
    assert heterogeneous.max() > 100


def test_topology_respects_packet_counts_and_pairs_target_profiles() -> None:
    probabilities = np.array([0.4, 0.55, 0.75, 0.85])
    balanced_counts = np.array([100, 100, 100])
    heterogeneous_counts = np.array([60, 100, 140])

    balanced = generate_synthetic_topology_with_packet_counts(
        probabilities,
        balanced_counts,
        n_gateways=3,
        seed=7,
    )
    heterogeneous = generate_synthetic_topology_with_packet_counts(
        probabilities,
        heterogeneous_counts,
        n_gateways=3,
        seed=7,
    )

    assert balanced.total_packets == 300
    assert heterogeneous.total_packets == 300
    assert np.array_equal(
        balanced.target_link_probabilities,
        heterogeneous.target_link_probabilities,
    )

    for sensor, expected in zip(heterogeneous.sensors, heterogeneous_counts.tolist()):
        incidence = heterogeneous.incidence_by_sensor[sensor]
        assert incidence.shape == (expected, 3)
        assert np.all(incidence.any(axis=1))
