"""Regression tests for exact threshold conversion and deterministic ties."""

import numpy as np

from lgc.exact.dp import solve_exact_min_links_generic
from lgc.greedy.fairness_aware import fairness_aware_greedy_generic
from lgc.greedy.multicover import multicover_greedy
from lgc.thresholds import minimum_required_count


def test_minimum_required_count_uses_exact_decimal_ceiling() -> None:
    assert minimum_required_count(10, 80) == 8
    assert minimum_required_count(10, 80.1) == 9
    assert minimum_required_count(3, 33.333) == 1
    assert minimum_required_count(3, 33.334) == 2
    assert minimum_required_count(0, 95) == 0


def test_greedy_ties_use_lexical_link_identifier() -> None:
    packet_matrix = np.array([[True, True]], dtype=bool)
    packet_sensor = np.array(["s1"])
    link_ids = ["s1→z", "s1→a"]
    sensor_total = {"s1": 1}

    proposed = fairness_aware_greedy_generic(
        packet_matrix,
        packet_sensor,
        sensor_total,
        link_ids,
        ["s1"],
        lam=1.0,
        k_max=1,
    )
    multicover = multicover_greedy(
        packet_matrix,
        packet_sensor,
        sensor_total,
        link_ids,
        ["s1"],
        P_min_pct=100,
        S_min_pct=100,
        k_max=1,
    )

    assert proposed.iloc[0]["link_added"] == "s1→a"
    assert multicover.iloc[0]["link_added"] == "s1→a"


def test_exact_dp_maximizes_coverage_within_minimum_budget() -> None:
    result = solve_exact_min_links_generic(
        P_min_pct=50,
        S_min_pct=50,
        sensors=["s1"],
        pattern_lookup_by_sensor={
            "s1": [
                (frozenset({"a"}), 2),
                (frozenset({"b"}), 4),
                (frozenset({"a", "b"}), 4),
            ]
        },
        sensor_total={"s1": 10},
        total_packets=10,
        sensor_gateway_lists={"s1": ["b", "a"]},
        max_budget=2,
    )

    assert result["status"] == "Optimal"
    assert result["n_links"] == 1
    assert result["selected_links"] == ["s1→b"]


def test_exact_dp_remaining_tie_is_canonical() -> None:
    result = solve_exact_min_links_generic(
        P_min_pct=50,
        S_min_pct=50,
        sensors=["s1"],
        pattern_lookup_by_sensor={
            "s1": [
                (frozenset({"a"}), 5),
                (frozenset({"b"}), 5),
            ]
        },
        sensor_total={"s1": 10},
        total_packets=10,
        sensor_gateway_lists={"s1": ["b", "a"]},
        max_budget=2,
    )

    assert result["selected_links"] == ["s1→a"]
