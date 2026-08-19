"""Exact sensor-separable dynamic program."""

from collections.abc import Iterable
from time import perf_counter

import numpy as np
import pandas as pd

from ..model import LinkGraph
from ..thresholds import minimum_required_count

def sensor_candidate_options_generic(
    sensor: str,
    S_min_pct: float,
    pattern_lookup_s: list[tuple[frozenset, int]],
    total_s: int,
    gateways: list[str],
) -> list[dict]:
    gateways = sorted(gateways)
    n_gw = len(gateways)
    required_sensor_packets = minimum_required_count(total_s, S_min_pct)
    options = []
    for mask in range(2**n_gw):
        subset = frozenset(gateways[i] for i in range(n_gw) if (mask >> i) & 1)
        covered = sum(cnt for gw_set, cnt in pattern_lookup_s if gw_set & subset)
        if covered >= required_sensor_packets:
            options.append({"cost": len(subset), "covered": covered, "links": subset})
    return options

def sensor_candidate_options(
    graph: LinkGraph, sensor: str, S_min_pct: float
) -> list[dict]:
    return sensor_candidate_options_generic(
        sensor=sensor,
        S_min_pct=S_min_pct,
        pattern_lookup_s=graph.pattern_lookup[sensor],
        total_s=graph.sensor_total[sensor],
        gateways=graph.sensor_gateway_lists[sensor],
    )

def solve_exact_min_links_generic(
    P_min_pct: float,
    S_min_pct: float,
    sensors: Iterable[str],
    pattern_lookup_by_sensor: dict[str, list[tuple[frozenset, int]]],
    sensor_total: dict[str, int],
    total_packets: int,
    sensor_gateway_lists: dict[str, list[str]],
    max_budget: int,
) -> dict:
    # deterministic reconstruction
    sensors = sorted(sensors)
    NEG = -1
    dp = np.full(max_budget + 1, NEG, dtype=np.int64)
    dp[0] = 0
    backptrs: list[dict[int, tuple[int, dict]]] = []

    for s in sensors:
        options = sensor_candidate_options_generic(
            sensor=s,
            S_min_pct=S_min_pct,
            pattern_lookup_s=pattern_lookup_by_sensor.get(s, []),
            total_s=sensor_total.get(s, 0),
            gateways=sensor_gateway_lists.get(s, []),
        )
        if not options:
            return {"status": "Infeasible", "n_links": None, "selected_links": None}

        new_dp = np.full(max_budget + 1, NEG, dtype=np.int64)
        choice: dict[int, tuple[int, dict]] = {}
        for prev_b in range(max_budget + 1):
            if dp[prev_b] == NEG:
                continue
            for opt in options:
                nb = prev_b + opt["cost"]
                if nb > max_budget:
                    continue
                candidate = dp[prev_b] + opt["covered"]
                # For a fixed budget, maximize aggregate covered packets
                if candidate > new_dp[nb]:
                    new_dp[nb] = candidate
                    choice[nb] = (prev_b, opt)
        dp = new_dp
        backptrs.append(choice)

    target = minimum_required_count(total_packets, P_min_pct)
    feasible = [b for b in range(max_budget + 1) if dp[b] >= target]
    if not feasible:
        return {"status": "Infeasible", "n_links": None, "selected_links": None}

    best_budget = min(feasible)
    selected: list[str] = []
    b = best_budget
    for i in range(len(sensors) - 1, -1, -1):
        s = sensors[i]
        prev_b, opt = backptrs[i][b]
        for g in opt["links"]:
            selected.append(f"{s}→{g}")
        b = prev_b

    return {"status": "Optimal", "n_links": best_budget, "selected_links": selected}

def solve_exact_min_links(graph: LinkGraph, P_min_pct: float, S_min_pct: float) -> dict:
    return solve_exact_min_links_generic(
        P_min_pct=P_min_pct,
        S_min_pct=S_min_pct,
        sensors=graph.all_sensors,
        pattern_lookup_by_sensor=graph.pattern_lookup,
        sensor_total=graph.sensor_total,
        total_packets=graph.total_unique_packets,
        sensor_gateway_lists=graph.sensor_gateway_lists,
        max_budget=graph.n_links,
    )

def benchmark_dp(
    graph: LinkGraph,
    requirements: Iterable[tuple[int, int]],
    n_reps: int = 100,
) -> pd.DataFrame:
    rows = []
    for P_min, S_min in requirements:
        elapsed = []
        for _ in range(n_reps):
            t = perf_counter()
            solve_exact_min_links(graph, P_min, S_min)
            elapsed.append(1000.0 * (perf_counter() - t))
        rows.append(
            {
                "P_min_pct": P_min,
                "S_min_pct": S_min,
                "n_reps": n_reps,
                "median_ms": float(np.median(elapsed)),
                "p95_ms": float(np.quantile(elapsed, 0.95)),
                "min_ms": float(np.min(elapsed)),
                "max_ms": float(np.max(elapsed)),
            }
        )
    return pd.DataFrame(rows)
