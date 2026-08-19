"""Meet-in-the-middle counter for the number of distinct optimum subsets."""

import bisect
from collections import defaultdict

import pandas as pd

from ..exact.dp import sensor_candidate_options
from ..model import LinkGraph
from ..thresholds import minimum_required_count

def _enumerate_half(
    sensor_names: list[str], F_by_sensor: dict[str, list[dict]]
) -> dict[int, list[int]]:
    combos: list[tuple[int, int]] = [(0, 0)]
    for s in sensor_names:
        opts = F_by_sensor[s]
        combos = [
            (b + opt["cost"], cov + opt["covered"])
            for (b, cov) in combos
            for opt in opts
        ]
    grouped: dict[int, list[int]] = defaultdict(list)
    for b, cov in combos:
        grouped[b].append(cov)
    return grouped

def count_optimal_subsets(
    sensors_list: list[str],
    F_by_sensor: dict[str, list[dict]],
    k_star: int,
    target_packets: int,
) -> int:
    half = len(sensors_list) // 2
    left = _enumerate_half(sensors_list[:half], F_by_sensor)
    right = _enumerate_half(sensors_list[half:], F_by_sensor)
    for b in right:
        right[b].sort()

    total = 0
    for b_left, covs_left in left.items():
        b_right = k_star - b_left
        if b_right < 0 or b_right not in right:
            continue
        sorted_right = right[b_right]
        for cov_left in covs_left:
            needed = target_packets - cov_left
            idx = bisect.bisect_left(sorted_right, needed)
            total += len(sorted_right) - idx
    return total

def build_multiplicity_table(
    graph: LinkGraph,
    requirements: list[tuple[int, int]],
    dp_results: dict[tuple[int, int], dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for P_min, S_min in requirements:
        dp_res = dp_results[(P_min, S_min)]
        if dp_res["status"] != "Optimal":
            continue
        k_star = dp_res["n_links"]
        target = minimum_required_count(graph.total_unique_packets, P_min)
        F_by_sensor = {
            s: sensor_candidate_options(graph, s, float(S_min))
            for s in graph.all_sensors
        }
        n_opts_per_sensor = {s: len(F_by_sensor[s]) for s in graph.all_sensors}
        n_opts = count_optimal_subsets(graph.all_sensors, F_by_sensor, k_star, target)
        rows.append(
            {
                "P_min_pct": P_min,
                "S_min_pct": S_min,
                "k_star": k_star,
                "n_optimal_subsets": n_opts,
                "n_options_per_sensor_min": min(n_opts_per_sensor.values()),
                "n_options_per_sensor_max": max(n_opts_per_sensor.values()),
                "note": "unique optimum" if n_opts == 1 else "multiple optima",
            }
        )
    return pd.DataFrame(rows)
