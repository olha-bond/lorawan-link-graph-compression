"""Diagnostic trace for fairness scoring with shared worst-sensor ties."""

import numpy as np
import pandas as pd


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


def fairness_worst_tie_diagnostic(
    lam: float = 1.0,
    k_max: int = 3,
) -> pd.DataFrame:
    packet_matrix, packet_sensor, sensor_total, link_ids, sensors = _shared_worst_case()

    covered = np.zeros(packet_matrix.shape[0], dtype=bool)
    sensor_covered = {sensor: 0 for sensor in sensors}
    remaining = sorted(range(packet_matrix.shape[1]), key=link_ids.__getitem__)
    rows: list[dict] = []

    def coverage_by_sensor(counts: dict[str, int]) -> dict[str, float]:
        return {
            sensor: 100 * counts[sensor] / sensor_total[sensor]
            for sensor in sensors
        }

    for step in range(min(k_max, packet_matrix.shape[1])):
        current_coverage = coverage_by_sensor(sensor_covered)
        current_worst = min(current_coverage.values())
        n_at_worst = sum(
            np.isclose(value, current_worst)
            for value in current_coverage.values()
        )

        candidates: list[dict] = []
        best_score = -np.inf
        best_col = None
        best_new_mask = None

        for column in remaining:
            new_mask = packet_matrix[:, column] & (~covered)
            n_new = int(new_mask.sum())
            packet_gain = 100 * n_new / packet_matrix.shape[0]

            temp = dict(sensor_covered)
            if n_new:
                sensors_new = packet_sensor[new_mask]
                uniq, counts = np.unique(sensors_new, return_counts=True)
                for sensor, count in zip(uniq, counts):
                    temp[sensor] = temp.get(sensor, 0) + int(count)

            candidate_coverage = coverage_by_sensor(temp)
            candidate_worst = min(candidate_coverage.values())
            fairness_gain = candidate_worst - current_worst
            score = packet_gain + lam * fairness_gain

            candidate = {
                "step": step + 1,
                "lambda": lam,
                "current_worst_sensor_coverage_pct": current_worst,
                "n_sensors_at_current_worst": n_at_worst,
                "candidate_link": link_ids[column],
                "new_packets": n_new,
                "packet_gain_pct": packet_gain,
                "candidate_worst_sensor_coverage_pct": candidate_worst,
                "fairness_gain_pct": fairness_gain,
                "marginal_score": score,
                "selected": False,
            }
            candidates.append(candidate)

            if score > best_score:
                best_score = score
                best_col = column
                best_new_mask = new_mask

        if best_col is None or best_new_mask is None:
            break

        selected_link = link_ids[best_col]
        for candidate in candidates:
            if candidate["candidate_link"] == selected_link:
                candidate["selected"] = True

        rows.extend(candidates)
        covered |= best_new_mask
        remaining.remove(best_col)

        if best_new_mask.any():
            sensors_new = packet_sensor[best_new_mask]
            uniq, counts = np.unique(sensors_new, return_counts=True)
            for sensor, count in zip(uniq, counts):
                sensor_covered[sensor] = sensor_covered.get(sensor, 0) + int(count)

    return pd.DataFrame(rows)
