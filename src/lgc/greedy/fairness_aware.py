"""Fairness-aware greedy link ordering."""

import numpy as np
import pandas as pd

from ..model import LinkGraph

def fairness_aware_greedy_generic(
    packet_matrix: np.ndarray,
    packet_sensor: np.ndarray,
    sensor_total: dict[str, int],
    link_ids: list[str],
    sensors_list: list[str],
    lam: float = 1.0,
    k_max: int | None = None,
) -> pd.DataFrame:
    n_packets, n_links = packet_matrix.shape
    k_max = k_max or n_links
    if not sensors_list or n_packets == 0:
        return pd.DataFrame()

    sensors_list = sorted(sensors_list)
    covered = np.zeros(n_packets, dtype=bool)
    sensor_covered = {s: 0 for s in sensors_list}
    # lexicographic tie-break
    remaining = sorted(range(n_links), key=link_ids.__getitem__)
    history = []

    def worst_and_median(counts: dict[str, int]) -> tuple[float, float]:
        pcts = [
            100 * counts[s] / sensor_total[s]
            for s in sensors_list
            if sensor_total[s] > 0
        ]
        if not pcts:
            return 0.0, 0.0
        return min(pcts), float(np.median(pcts))

    current_worst, _ = worst_and_median(sensor_covered)

    for step in range(k_max):
        best_score, best_col, best_new_mask = -np.inf, None, None

        for j in remaining:
            new_mask = packet_matrix[:, j] & (~covered)
            n_new = new_mask.sum()
            gain_packet = 100 * n_new / n_packets

            if n_new:
                sensors_new = packet_sensor[new_mask]
                uniq, cnts = np.unique(sensors_new, return_counts=True)
                temp = dict(sensor_covered)
                for s, c in zip(uniq, cnts):
                    temp[s] = temp.get(s, 0) + c
            else:
                temp = sensor_covered

            new_worst, _ = worst_and_median(temp)
            score = gain_packet + lam * (new_worst - current_worst)

            if score > best_score:
                best_score, best_col, best_new_mask = score, j, new_mask

        if best_col is None:
            break

        covered |= best_new_mask
        remaining.remove(best_col)

        if best_new_mask.any():
            sensors_new = packet_sensor[best_new_mask]
            uniq, cnts = np.unique(sensors_new, return_counts=True)
            for s, c in zip(uniq, cnts):
                sensor_covered[s] = sensor_covered.get(s, 0) + c

        current_worst, current_median = worst_and_median(sensor_covered)
        history.append(
            {
                "step": step + 1,
                "link_added": link_ids[best_col],
                "k_links": step + 1,
                "packet_coverage_pct": 100 * covered.sum() / n_packets,
                "worst_sensor_coverage_pct": current_worst,
                "median_sensor_coverage_pct": current_median,
                "marginal_score": best_score,
            }
        )

    return pd.DataFrame(history)

def fairness_aware_greedy(
    graph: LinkGraph, lam: float = 1.0, k_max: int | None = None
) -> pd.DataFrame:
    return fairness_aware_greedy_generic(
        packet_matrix=graph.incidence,
        packet_sensor=graph.packet_sensor,
        sensor_total=graph.sensor_total,
        link_ids=graph.all_link_ids,
        sensors_list=graph.all_sensors,
        lam=lam,
        k_max=k_max,
    )
