"""Threshold-dependent multi-submodular cover greedy."""

import numpy as np
import pandas as pd

from ..model import LinkGraph
from ..thresholds import minimum_required_count

def multicover_greedy(
    packet_matrix: np.ndarray,
    packet_sensor: np.ndarray,
    sensor_total: dict[str, int],
    link_ids: list[str],
    sensors_list: list[str],
    P_min_pct: float,
    S_min_pct: float,
    k_max: int | None = None,
) -> pd.DataFrame:
    n_packets, n_links = packet_matrix.shape
    k_max = k_max or n_links
    if not sensors_list or n_packets == 0:
        return pd.DataFrame()

    sensors_list = sorted(sensors_list)
    # fractional cap for gains, integer stop for feasibility
    t_agg = (P_min_pct / 100.0) * n_packets
    t_sensor = {
        s: (S_min_pct / 100.0) * sensor_total.get(s, 0)
        for s in sensors_list
    }
    required_agg = minimum_required_count(n_packets, P_min_pct)
    required_sensor = {
        s: minimum_required_count(sensor_total.get(s, 0), S_min_pct)
        for s in sensors_list
    }

    def capped_F(agg_count: int, counts: dict[str, int]) -> float:
        f = min(agg_count / t_agg, 1.0) if t_agg > 0 else 1.0
        for s in sensors_list:
            ts = t_sensor[s]
            f += min(counts[s] / ts, 1.0) if ts > 0 else 1.0
        return f

    covered = np.zeros(n_packets, dtype=bool)
    sensor_covered = {s: 0 for s in sensors_list}
    remaining = sorted(range(n_links), key=link_ids.__getitem__)
    history = []
    current_F = capped_F(0, sensor_covered)

    for step in range(k_max):
        best_gain, best_col, best_new_mask = -np.inf, None, None

        for j in remaining:
            new_mask = packet_matrix[:, j] & (~covered)
            n_new = int(new_mask.sum())
            new_agg = int(covered.sum()) + n_new

            if n_new:
                sensors_new = packet_sensor[new_mask]
                uniq, cnts = np.unique(sensors_new, return_counts=True)
                temp = dict(sensor_covered)
                for s, c in zip(uniq, cnts):
                    temp[s] = temp.get(s, 0) + int(c)
            else:
                temp = sensor_covered

            new_F = capped_F(new_agg, temp)
            gain = new_F - current_F
            if gain > best_gain:
                best_gain, best_col, best_new_mask = gain, j, new_mask

        if best_col is None:
            break

        covered |= best_new_mask
        remaining.remove(best_col)

        if best_new_mask.any():
            sensors_new = packet_sensor[best_new_mask]
            uniq, cnts = np.unique(sensors_new, return_counts=True)
            for s, c in zip(uniq, cnts):
                sensor_covered[s] = sensor_covered.get(s, 0) + int(c)

        current_F = capped_F(int(covered.sum()), sensor_covered)
        pcts = [
            100 * sensor_covered[s] / sensor_total[s]
            for s in sensors_list
            if sensor_total.get(s, 0) > 0
        ]
        worst = min(pcts) if pcts else 0.0
        median = float(np.median(pcts)) if pcts else 0.0

        history.append(
            {
                "step": step + 1,
                "link_added": link_ids[best_col],
                "k_links": step + 1,
                "packet_coverage_pct": 100 * covered.sum() / n_packets,
                "worst_sensor_coverage_pct": worst,
                "median_sensor_coverage_pct": median,
                "marginal_score": best_gain,
                "F_total": current_F,
            }
        )

        aggregate_covered = int(covered.sum())
        if aggregate_covered >= required_agg and all(
            sensor_covered[s] >= required_sensor[s] for s in sensors_list
        ):
            break

    return pd.DataFrame(history)

def multicover_greedy_fast(
    graph: LinkGraph,
    P_min_pct: float,
    S_min_pct: float,
    k_max: int | None = None,
) -> pd.DataFrame:
    n_packets = graph.incidence.shape[0]
    k_max = k_max or graph.n_links

    t_agg = (P_min_pct / 100.0) * n_packets
    t_sensor = {
        s: (S_min_pct / 100.0) * graph.sensor_total[s]
        for s in graph.all_sensors
    }
    required_agg = minimum_required_count(n_packets, P_min_pct)
    required_sensor = {
        s: minimum_required_count(graph.sensor_total[s], S_min_pct)
        for s in graph.all_sensors
    }

    covered_local = {
        s: np.zeros(graph.sensor_packet_count[s], dtype=bool) for s in graph.all_sensors
    }
    sensor_covered = {s: 0 for s in graph.all_sensors}
    agg_count = 0
    remaining = sorted(graph.all_link_ids)
    history = []

    for step in range(k_max):
        best_gain = -np.inf
        best_link = None
        best_new_mask = None
        best_sensor = None
        best_n_new = 0

        for link in remaining:
            s = graph.link_sensor[link]
            new_mask = graph.link_local_mask[link] & (~covered_local[s])
            n_new = int(new_mask.sum())

            if n_new == 0:
                gain = 0.0
            else:
                new_agg = agg_count + n_new
                new_scount = sensor_covered[s] + n_new

                agg_gain = (
                    min(new_agg / t_agg, 1.0) - min(agg_count / t_agg, 1.0)
                    if t_agg > 0
                    else 0.0
                )
                sensor_gain = (
                    min(new_scount / t_sensor[s], 1.0)
                    - min(sensor_covered[s] / t_sensor[s], 1.0)
                    if t_sensor[s] > 0
                    else 0.0
                )
                gain = agg_gain + sensor_gain

            if gain > best_gain:
                best_gain = gain
                best_link = link
                best_new_mask = new_mask
                best_sensor = s
                best_n_new = n_new

        if best_link is None:
            break

        covered_local[best_sensor] |= best_new_mask
        sensor_covered[best_sensor] += best_n_new
        agg_count += best_n_new
        remaining.remove(best_link)

        pcts = [
            100 * sensor_covered[s] / graph.sensor_total[s]
            for s in graph.all_sensors
            if graph.sensor_total[s] > 0
        ]
        worst = min(pcts) if pcts else 0.0
        median = float(np.median(pcts)) if pcts else 0.0

        F_agg = min(agg_count / t_agg, 1.0) if t_agg > 0 else 1.0
        F_sensors = sum(
            (min(sensor_covered[s] / t_sensor[s], 1.0) if t_sensor[s] > 0 else 1.0)
            for s in graph.all_sensors
        )
        current_F = F_agg + F_sensors

        history.append(
            {
                "step": step + 1,
                "link_added": best_link,
                "k_links": step + 1,
                "packet_coverage_pct": 100 * agg_count / n_packets,
                "worst_sensor_coverage_pct": worst,
                "median_sensor_coverage_pct": median,
                "marginal_score": best_gain,
                "F_total": current_F,
            }
        )

        if agg_count >= required_agg and all(
            sensor_covered[s] >= required_sensor[s] for s in graph.all_sensors
        ):
            break

    return pd.DataFrame(history)
