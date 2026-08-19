"""Link selection and month-level evaluation for temporal windows."""

from collections.abc import Callable, Iterable
from hashlib import sha256

import numpy as np

from ...exact.dp import solve_exact_min_links_generic
from ...model import LinkGraph
from ...thresholds import minimum_required_count
from .context import combine_window
from .models import CombinedWindow, FitResult, SubsetEvaluation, TemporalContext

GainFunction = Callable[[str, int, int, dict[str, int], float], float]


def _fit_exact(
    graph: LinkGraph,
    window: CombinedWindow,
    P_min: float,
    S_min: float,
) -> FitResult:
    if not window.sensors_active:
        return FitResult("Infeasible", None, [])

    result = solve_exact_min_links_generic(
        P_min_pct=P_min,
        S_min_pct=S_min,
        sensors=window.sensors_active,
        pattern_lookup_by_sensor=window.pattern_lookup,
        sensor_total=window.sensor_total,
        total_packets=window.total_packets,
        sensor_gateway_lists=graph.sensor_gateway_lists,
        max_budget=graph.n_links,
    )
    return FitResult.from_mapping(result)


def _sensor_positions(window: CombinedWindow) -> dict[str, np.ndarray]:
    return {
        sensor: np.flatnonzero(window.packet_sensor == sensor)
        for sensor in window.sensors_active
    }


def _worst_sensor_coverage(
    window: CombinedWindow,
    counts: dict[str, int],
) -> float:
    return min(
        100.0 * counts[sensor] / window.sensor_total[sensor]
        for sensor in window.sensors_active
    )


def _run_greedy(
    graph: LinkGraph,
    window: CombinedWindow,
    P_min: float,
    S_min: float,
    gain_function: GainFunction,
) -> FitResult:
    sensors = window.sensors_active
    if not sensors:
        return FitResult("Infeasible", None, [])

    positions = _sensor_positions(window)
    covered = {
        sensor: np.zeros(len(positions[sensor]), dtype=bool) for sensor in sensors
    }
    sensor_covered = {sensor: 0 for sensor in sensors}
    remaining = sorted(graph.all_link_ids)
    selected: list[str] = []
    aggregate_covered = 0
    aggregate_target = minimum_required_count(window.total_packets, P_min)
    sensor_targets = {
        sensor: minimum_required_count(window.sensor_total[sensor], S_min)
        for sensor in sensors
    }

    for _ in range(graph.n_links):
        current_worst = _worst_sensor_coverage(window, sensor_covered)
        best_gain = -np.inf
        best_link: str | None = None
        best_sensor: str | None = None
        best_new_mask: np.ndarray | None = None
        best_n_new = 0

        for link in remaining:
            sensor = graph.link_sensor[link]
            if sensor not in positions:
                new_mask = None
                n_new = 0
            else:
                column = graph.link_to_col[link]
                link_mask = window.packet_matrix[positions[sensor], column]
                new_mask = link_mask & ~covered[sensor]
                n_new = int(new_mask.sum())

            gain = gain_function(
                sensor,
                n_new,
                aggregate_covered,
                sensor_covered,
                current_worst,
            )
            if gain > best_gain:
                best_gain = gain
                best_link = link
                best_sensor = sensor
                best_new_mask = new_mask
                best_n_new = n_new

        if best_link is None:
            break

        selected.append(best_link)
        remaining.remove(best_link)
        if best_sensor in covered and best_new_mask is not None:
            covered[best_sensor] |= best_new_mask
            sensor_covered[best_sensor] += best_n_new
            aggregate_covered += best_n_new

        if aggregate_covered >= aggregate_target and all(
            sensor_covered[sensor] >= sensor_targets[sensor] for sensor in sensors
        ):
            return FitResult("Feasible", len(selected), sorted(selected))

    return FitResult("Infeasible", None, [])


def _fit_proposed(
    graph: LinkGraph,
    window: CombinedWindow,
    P_min: float,
    S_min: float,
    lam: float,
) -> FitResult:
    def gain(
        sensor: str,
        n_new: int,
        _aggregate_covered: int,
        sensor_covered: dict[str, int],
        current_worst: float,
    ) -> float:
        if n_new == 0 or sensor not in sensor_covered:
            new_worst = current_worst
        else:
            updated = dict(sensor_covered)
            updated[sensor] += n_new
            new_worst = _worst_sensor_coverage(window, updated)
        return (
            100.0 * n_new / window.total_packets
            + lam * (new_worst - current_worst)
        )

    return _run_greedy(graph, window, P_min, S_min, gain)


def _fit_multicover(
    graph: LinkGraph,
    window: CombinedWindow,
    P_min: float,
    S_min: float,
) -> FitResult:
    # fractional caps for gains, integers for feasibility
    target_aggregate = P_min / 100.0 * window.total_packets
    target_sensor = {
        sensor: S_min / 100.0 * window.sensor_total[sensor]
        for sensor in window.sensors_active
    }

    def gain(
        sensor: str,
        n_new: int,
        aggregate_covered: int,
        sensor_covered: dict[str, int],
        _current_worst: float,
    ) -> float:
        if n_new == 0 or sensor not in sensor_covered:
            return 0.0

        aggregate_gain = (
            min((aggregate_covered + n_new) / target_aggregate, 1.0)
            - min(aggregate_covered / target_aggregate, 1.0)
            if target_aggregate > 0
            else 0.0
        )
        sensor_gain = (
            min((sensor_covered[sensor] + n_new) / target_sensor[sensor], 1.0)
            - min(sensor_covered[sensor] / target_sensor[sensor], 1.0)
            if target_sensor[sensor] > 0
            else 0.0
        )
        return aggregate_gain + sensor_gain

    return _run_greedy(graph, window, P_min, S_min, gain)


def fit_from_combined_window(
    graph: LinkGraph,
    window: CombinedWindow,
    P_min: float,
    S_min: float,
    method: str,
    lam: float = 1.0,
) -> FitResult:
    if method == "exact":
        return _fit_exact(graph, window, P_min, S_min)
    if method == "proposed":
        return _fit_proposed(graph, window, P_min, S_min, lam)
    if method == "multicover":
        return _fit_multicover(graph, window, P_min, S_min)
    if method == "coverage_only":
        return _fit_proposed(graph, window, P_min, S_min, lam=0.0)
    raise ValueError(f"Unknown temporal fit method: {method}")


def fit_on_window(
    graph: LinkGraph,
    context: TemporalContext,
    months: Iterable[str],
    P_min: float,
    S_min: float,
    method: str,
    lam: float = 1.0,
) -> FitResult:
    window = combine_window(graph, context, months)
    return fit_from_combined_window(
        graph,
        window,
        P_min,
        S_min,
        method=method,
        lam=lam,
    )


def evaluate_subset_on_month(
    graph: LinkGraph,
    context: TemporalContext,
    selected_links: Iterable[str],
    month: str,
) -> SubsetEvaluation:
    rows = context.packet_rows_by_month[month]
    n_packets = len(rows)
    if n_packets == 0:
        return SubsetEvaluation(float("nan"), float("nan"), float("nan"), 0)

    selected = [link for link in selected_links if link in graph.link_to_col]
    if selected:
        cols = [graph.link_to_col[link] for link in selected]
        covered = graph.incidence[np.ix_(rows, cols)].any(axis=1)
    else:
        covered = np.zeros(n_packets, dtype=bool)

    month_sensors = graph.packet_sensor[rows]
    sensor_coverages = []
    for sensor in sorted(set(month_sensors.tolist())):
        sensor_mask = month_sensors == sensor
        sensor_coverages.append(
            100.0 * covered[sensor_mask].sum() / sensor_mask.sum()
        )

    return SubsetEvaluation(
        packet_coverage_pct=100.0 * covered.sum() / n_packets,
        worst_sensor_coverage_pct=(
            min(sensor_coverages) if sensor_coverages else float("nan")
        ),
        median_sensor_coverage_pct=(
            float(np.median(sensor_coverages))
            if sensor_coverages
            else float("nan")
        ),
        n_active_sensors=len(sensor_coverages),
    )


def subset_hash(selected_links: Iterable[str]) -> str:
    canonical = ";".join(sorted(selected_links))
    return sha256(canonical.encode("utf-8")).hexdigest()[:16]
