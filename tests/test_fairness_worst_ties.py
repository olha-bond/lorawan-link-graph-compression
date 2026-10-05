"""Regression tests for fairness scoring with shared worst-sensor ties."""

import numpy as np

from lgc.greedy.fairness_aware import fairness_aware_greedy_generic


def _shared_worst_case() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, int],
    list[str],
    list[str],
]:
    packet_matrix = np.zeros((9, 4), dtype=bool)
    packet_matrix[0:4, 0] = True
    packet_matrix[5:7, 1] = True
    packet_matrix[7, 2] = True
    packet_matrix[4, 3] = True

    packet_sensor = np.array(
        ["s1", "s1", "s1", "s1", "s1", "s2", "s2", "s3", "s3"]
    )
    sensor_total = {"s1": 5, "s2": 2, "s3": 2}
    link_ids = ["s1→a", "s2→a", "s3→a", "s1→b"]
    sensors = ["s1", "s2", "s3"]

    return packet_matrix, packet_sensor, sensor_total, link_ids, sensors


def test_fairness_gain_is_zero_while_multiple_sensors_share_zero_minimum() -> None:
    packet_matrix, packet_sensor, sensor_total, link_ids, sensors = _shared_worst_case()

    no_fairness = fairness_aware_greedy_generic(
        packet_matrix,
        packet_sensor,
        sensor_total,
        link_ids,
        sensors,
        lam=0.0,
        k_max=2,
    )
    fairness = fairness_aware_greedy_generic(
        packet_matrix,
        packet_sensor,
        sensor_total,
        link_ids,
        sensors,
        lam=1.0,
        k_max=2,
    )

    assert no_fairness["link_added"].tolist() == ["s1→a", "s2→a"]
    assert fairness["link_added"].tolist() == ["s1→a", "s2→a"]
    assert np.allclose(
        no_fairness["marginal_score"].to_numpy(),
        fairness["marginal_score"].to_numpy(),
    )
    assert fairness["worst_sensor_coverage_pct"].tolist() == [0.0, 0.0]


def test_fairness_gain_activates_when_candidate_lifts_unique_worst_sensor() -> None:
    packet_matrix, packet_sensor, sensor_total, link_ids, sensors = _shared_worst_case()

    no_fairness = fairness_aware_greedy_generic(
        packet_matrix,
        packet_sensor,
        sensor_total,
        link_ids,
        sensors,
        lam=0.0,
        k_max=3,
    )
    fairness = fairness_aware_greedy_generic(
        packet_matrix,
        packet_sensor,
        sensor_total,
        link_ids,
        sensors,
        lam=1.0,
        k_max=3,
    )

    assert no_fairness.iloc[2]["link_added"] == "s1→b"
    assert fairness.iloc[2]["link_added"] == "s3→a"
    assert fairness.iloc[2]["worst_sensor_coverage_pct"] == 50.0
    assert fairness.iloc[2]["marginal_score"] > no_fairness.iloc[2]["marginal_score"]
