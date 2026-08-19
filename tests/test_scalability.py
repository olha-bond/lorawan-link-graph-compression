"""Tests for the UVA-calibrated synthetic scalability helpers."""

import numpy as np
import pandas as pd

from lgc.analysis.scalability import (
    empirical_link_coverage_profiles,
    fairness_aware_prefix_benchmark,
    generate_synthetic_topology,
)
from lgc.greedy.fairness_aware import fairness_aware_greedy_generic
from lgc.metrics import min_k_for_requirement


def test_empirical_link_coverage_profiles_uses_observed_packet_denominator() -> None:
    df = pd.DataFrame(
        {
            "sensor": ["s1", "s1", "s1", "s1"],
            "gateway": ["g1", "g2", "g1", "g1"],
            "counter": [1, 1, 2, 3],
        }
    )
    out = empirical_link_coverage_profiles(df)
    got = {
        (row.sensor, row.gateway): row.coverage_fraction
        for row in out.itertuples(index=False)
    }
    assert got[("s1", "g1")] == 1.0
    assert got[("s1", "g2")] == 1.0 / 3.0


def test_synthetic_generator_produces_nonempty_observed_packets() -> None:
    topology = generate_synthetic_topology(
        empirical_probabilities=np.array([0.35, 0.55, 0.8]),
        n_sensors=4,
        n_gateways=3,
        packets_per_sensor=120,
        seed=7,
    )
    assert topology.n_links == 12
    assert topology.total_packets == 480
    for incidence in topology.incidence_by_sensor.values():
        assert incidence.shape == (120, 3)
        assert np.all(incidence.any(axis=1))


def test_fast_scalability_prefix_matches_publication_greedy_on_small_graph() -> None:
    requirements = [(80, 60), (90, 70)]
    topology = generate_synthetic_topology(
        empirical_probabilities=np.array([0.4, 0.55, 0.75, 0.85]),
        n_sensors=3,
        n_gateways=3,
        packets_per_sensor=80,
        seed=3,
    )

    sensors = topology.sensors
    link_ids = []
    columns = []
    packet_sensor = []
    for s in sensors:
        inc = topology.incidence_by_sensor[s]
        for g in range(inc.shape[1]):
            col = np.zeros(topology.total_packets, dtype=bool)
            offset = sensors.index(s) * inc.shape[0]
            col[offset : offset + inc.shape[0]] = inc[:, g]
            columns.append(col)
            link_ids.append(f"{s}→g{g + 1:02d}")
        packet_sensor.extend([s] * inc.shape[0])

    packet_matrix = np.column_stack(columns)
    generic = fairness_aware_greedy_generic(
        packet_matrix=packet_matrix,
        packet_sensor=np.asarray(packet_sensor),
        sensor_total=topology.sensor_total,
        link_ids=link_ids,
        sensors_list=sensors,
        lam=1.0,
    )

    fast_k, _ = fairness_aware_prefix_benchmark(topology, requirements, lam=1.0)
    for req in requirements:
        expected = min_k_for_requirement(generic, req[0], req[1])
        assert fast_k[req] == expected
