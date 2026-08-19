import pandas as pd

from lgc.analysis.temporal import (
    build_temporal_context,
    combine_window,
    evaluate_subset_on_month,
    fit_on_window,
    run_rolling_prospective,
    solve_exact_all_month_static,
)
from lgc.greedy.fairness_aware import fairness_aware_greedy_generic
from lgc.greedy.multicover import multicover_greedy
from lgc.metrics import min_k_for_requirement, selected_up_to
from lgc.model import LinkGraph


def _toy_graph():
    rows = [
        {"sensor": "s1", "gateway": "A", "counter": 1, "month": "2024-01"},
        {"sensor": "s1", "gateway": "B", "counter": 2, "month": "2024-01"},
        {"sensor": "s2", "gateway": "A", "counter": 1, "month": "2024-02"},
        {"sensor": "s2", "gateway": "B", "counter": 2, "month": "2024-02"},
        {"sensor": "s1", "gateway": "A", "counter": 3, "month": "2024-03"},
        {"sensor": "s2", "gateway": "B", "counter": 3, "month": "2024-03"},
    ]
    df = pd.DataFrame(rows)
    df["link_id"] = df["sensor"] + "→" + df["gateway"]
    df["rssi"] = -90.0
    df["snr"] = 5.0
    return LinkGraph(df)


def test_prospective_new_sensor_cold_start_has_zero_worst_coverage():
    graph = _toy_graph()
    context = build_temporal_context(graph)
    fit = fit_on_window(
        graph,
        context,
        ["2024-01"],
        P_min=50,
        S_min=50,
        method="proposed",
    )
    evaluation = evaluate_subset_on_month(
        graph,
        context,
        fit.selected_links,
        "2024-02",
    )
    assert evaluation.worst_sensor_coverage_pct == 0.0


def test_all_month_static_exact_selects_links_for_both_sensors():
    graph = _toy_graph()
    context = build_temporal_context(graph)
    result = solve_exact_all_month_static(graph, context, P_min=50, S_min=50)
    assert result.status == "Optimal"
    assert result.n_links == 2
    assert {link.split("→")[0] for link in result.selected_links} == {"s1", "s2"}


def test_rolling_output_contains_frozen_previous_expanding_and_oracle():
    graph = _toy_graph()
    context = build_temporal_context(graph)
    rolling = run_rolling_prospective(
        graph,
        context,
        requirements=[(50, 50)],
        frozen_initial_months=1,
    )
    policies = set(zip(rolling["policy"], rolling["fit_method"]))
    assert ("frozen_initial_window", "proposed") in policies
    assert ("previous_month", "proposed") in policies
    assert ("expanding_window", "multicover") in policies
    assert ("expanding_window", "coverage_only") in policies
    assert ("frozen_initial_window", "coverage_only") in policies
    assert ("retrospective_oracle", "exact") in policies


def test_fast_temporal_fit_matches_generic_greedy_implementations():
    graph = _toy_graph()
    context = build_temporal_context(graph)
    window = combine_window(graph, context, ["2024-03"])

    fast_proposed = fit_on_window(
        graph,
        context,
        ["2024-03"],
        50,
        50,
        method="proposed",
    )
    generic_proposed = fairness_aware_greedy_generic(
        packet_matrix=window.packet_matrix,
        packet_sensor=window.packet_sensor,
        sensor_total=window.sensor_total,
        link_ids=graph.all_link_ids,
        sensors_list=window.sensors_active,
        lam=1.0,
    )
    proposed_k = min_k_for_requirement(generic_proposed, 50, 50)
    assert fast_proposed.n_links == proposed_k
    assert fast_proposed.selected_links == selected_up_to(generic_proposed, proposed_k)

    fast_multicover = fit_on_window(
        graph,
        context,
        ["2024-03"],
        50,
        50,
        method="multicover",
    )
    generic_multicover = multicover_greedy(
        packet_matrix=window.packet_matrix,
        packet_sensor=window.packet_sensor,
        sensor_total=window.sensor_total,
        link_ids=graph.all_link_ids,
        sensors_list=window.sensors_active,
        P_min_pct=50,
        S_min_pct=50,
    )
    multicover_k = min_k_for_requirement(generic_multicover, 50, 50)
    assert fast_multicover.n_links == multicover_k
    assert fast_multicover.selected_links == selected_up_to(
        generic_multicover,
        multicover_k,
    )
