"""Synthetic topology scalability helpers calibrated to UVA link coverage."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from ..exact.dp import sensor_candidate_options_generic, solve_exact_min_links_generic
from ..thresholds import minimum_required_count


@dataclass
class SyntheticTopology:
    """Compact sensor-local representation of a synthetic link graph."""

    incidence_by_sensor: dict[str, np.ndarray]
    pattern_lookup_by_sensor: dict[str, list[tuple[frozenset[str], int]]]
    sensor_total: dict[str, int]
    sensor_gateway_lists: dict[str, list[str]]
    target_link_probabilities: np.ndarray
    realized_link_probabilities: np.ndarray

    @property
    def sensors(self) -> list[str]:
        return sorted(self.incidence_by_sensor)

    @property
    def n_links(self) -> int:
        return sum(len(v) for v in self.sensor_gateway_lists.values())

    @property
    def total_packets(self) -> int:
        return sum(self.sensor_total.values())


def empirical_link_coverage_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Return observed-packet coverage fractions for every sensor--gateway link."""
    packets = df[["sensor", "counter"]].drop_duplicates()
    sensor_total = packets.groupby("sensor").size().rename("sensor_packets")

    link_packets = (
        df[["sensor", "gateway", "counter"]]
        .drop_duplicates()
        .groupby(["sensor", "gateway"])
        .size()
        .rename("link_packets")
        .reset_index()
    )
    result = link_packets.merge(sensor_total.reset_index(), on="sensor", how="left")
    result["coverage_fraction"] = result["link_packets"] / result["sensor_packets"]
    return result.sort_values(["sensor", "gateway"]).reset_index(drop=True)


def _calibrate_unconditional_probabilities(
    target_conditional: np.ndarray,
) -> np.ndarray:
    """Bernoulli rates that reproduce the target marginals after zero rejection."""
    p = np.asarray(target_conditional, dtype=float)

    def f(d: float) -> float:
        return 1.0 - float(np.prod(1.0 - d * p)) - d

    lo = 1e-12
    hi = 1.0
    if f(hi) > 1e-12:
        raise RuntimeError("failed to bracket calibration root")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    d = 0.5 * (lo + hi)
    return d * p


def _sample_conditional_profile(
    pool: np.ndarray,
    n_gateways: int,
    rng: np.random.Generator,
    max_attempts: int = 10_000,
) -> np.ndarray:
    for _ in range(max_attempts):
        target = rng.choice(pool, size=n_gateways, replace=True).astype(float)
        if target.sum() > 1.0 + 1e-6:
            return target
    raise RuntimeError(
        "could not sample a valid conditional coverage profile; "
        "check the empirical probability pool"
    )


def _generate_observed_incidence(
    target_conditional: np.ndarray,
    n_observed_packets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    q = _calibrate_unconditional_probabilities(target_conditional)
    accepted: list[np.ndarray] = []
    n_have = 0

    while n_have < n_observed_packets:
        batch_n = max(256, 2 * (n_observed_packets - n_have))
        batch = rng.random((batch_n, len(q))) < q
        batch = batch[batch.any(axis=1)]
        if len(batch) == 0:
            continue
        take = min(len(batch), n_observed_packets - n_have)
        accepted.append(batch[:take])
        n_have += take

    return np.vstack(accepted)


def _pattern_lookup_from_incidence(
    incidence: np.ndarray,
    gateway_names: list[str],
) -> list[tuple[frozenset[str], int]]:
    weights = (1 << np.arange(incidence.shape[1], dtype=np.uint64))
    masks = incidence.astype(np.uint64) @ weights
    unique, counts = np.unique(masks, return_counts=True)

    lookup: list[tuple[frozenset[str], int]] = []
    for mask, count in zip(unique.tolist(), counts.tolist()):
        gateways = frozenset(
            gateway_names[j]
            for j in range(len(gateway_names))
            if int(mask) & (1 << j)
        )
        lookup.append((gateways, int(count)))
    return lookup


def generate_synthetic_topology(
    empirical_probabilities: np.ndarray,
    n_sensors: int,
    n_gateways: int,
    packets_per_sensor: int,
    seed: int,
) -> SyntheticTopology:
    """Generate a topology from UVA-calibrated link marginals."""
    if n_sensors <= 0 or n_gateways <= 0 or packets_per_sensor <= 0:
        raise ValueError("topology dimensions and packet count must be positive")

    pool = np.asarray(empirical_probabilities, dtype=float)
    pool = pool[np.isfinite(pool)]
    if len(pool) == 0:
        raise ValueError("empirical probability pool is empty")

    rng = np.random.default_rng(seed)
    gateway_names = [f"g{j + 1:02d}" for j in range(n_gateways)]

    incidence_by_sensor: dict[str, np.ndarray] = {}
    pattern_lookup_by_sensor: dict[str, list[tuple[frozenset[str], int]]] = {}
    sensor_total: dict[str, int] = {}
    sensor_gateway_lists: dict[str, list[str]] = {}
    targets: list[np.ndarray] = []
    realized: list[np.ndarray] = []

    for i in range(n_sensors):
        sensor = f"s{i + 1:04d}"
        target = _sample_conditional_profile(pool, n_gateways, rng)
        incidence = _generate_observed_incidence(target, packets_per_sensor, rng)

        incidence_by_sensor[sensor] = incidence
        pattern_lookup_by_sensor[sensor] = _pattern_lookup_from_incidence(
            incidence, gateway_names
        )
        sensor_total[sensor] = packets_per_sensor
        sensor_gateway_lists[sensor] = list(gateway_names)
        targets.append(target)
        realized.append(incidence.mean(axis=0))

    return SyntheticTopology(
        incidence_by_sensor=incidence_by_sensor,
        pattern_lookup_by_sensor=pattern_lookup_by_sensor,
        sensor_total=sensor_total,
        sensor_gateway_lists=sensor_gateway_lists,
        target_link_probabilities=np.concatenate(targets),
        realized_link_probabilities=np.concatenate(realized),
    )


def fairness_aware_prefix_benchmark(
    topology: SyntheticTopology,
    requirements: list[tuple[int, int]],
    lam: float = 1.0,
) -> tuple[dict, float]:
    """Build the reusable ordering until all requested nested requirements are met."""
    sensors = topology.sensors
    if not sensors:
        return {}, 0.0

    req_targets = {}
    for req in requirements:
        agg_target = minimum_required_count(topology.total_packets, req[0])
        sensor_targets = {
            s: minimum_required_count(topology.sensor_total[s], req[1])
            for s in sensors
        }
        req_targets[req] = (agg_target, sensor_targets)

    covered_local = {
        s: np.zeros(topology.sensor_total[s], dtype=bool) for s in sensors
    }
    sensor_covered = {s: 0 for s in sensors}
    remaining_gateways = {
        s: list(range(topology.incidence_by_sensor[s].shape[1])) for s in sensors
    }

    best_gateway: dict[str, int | None] = {}
    best_gain: dict[str, int] = {}

    def recompute_local_best(s: str) -> None:
        remaining = remaining_gateways[s]
        if not remaining:
            best_gateway[s] = None
            best_gain[s] = 0
            return

        incidence = topology.incidence_by_sensor[s]
        uncovered = ~covered_local[s]
        gains = (incidence[:, remaining] & uncovered[:, None]).sum(axis=0)

        max_gain = int(gains.max())
        pos = int(np.flatnonzero(gains == max_gain)[0])

        best_gateway[s] = remaining[pos]
        best_gain[s] = max_gain

    for s in sensors:
        recompute_local_best(s)

    selected_k: dict[tuple[int, int], int] = {}
    aggregate_covered = 0
    t0 = perf_counter()

    for step in range(1, topology.n_links + 1):
        pcts = np.array(
            [
                100.0 * sensor_covered[s] / topology.sensor_total[s]
                for s in sensors
            ],
            dtype=float,
        )

        order = np.argsort(pcts, kind="stable")

        min_idx = int(order[0])
        min_value = float(pcts[min_idx])
        min_count = int(np.sum(pcts == min_value))
        second_value = (
            float(pcts[order[1]]) if len(order) > 1 else float("inf")
        )

        chosen_sensor = None
        chosen_gateway = None
        chosen_score = -np.inf

        for idx, s in enumerate(sensors):
            g = best_gateway[s]
            if g is None:
                continue

            n_new = best_gain[s]
            new_pct = (
                100.0
                * (sensor_covered[s] + n_new)
                / topology.sensor_total[s]
            )

            if idx != min_idx or min_count > 1:
                other_min = min_value
            else:
                other_min = second_value

            new_worst = min(new_pct, other_min)
            score = (
                100.0 * n_new / topology.total_packets
                + lam * (new_worst - min_value)
            )

            if score > chosen_score:
                chosen_score = score
                chosen_sensor = s
                chosen_gateway = g

        if chosen_sensor is None or chosen_gateway is None:
            break

        s = chosen_sensor
        g = chosen_gateway

        incidence = topology.incidence_by_sensor[s]
        new_mask = incidence[:, g] & (~covered_local[s])
        n_new = int(new_mask.sum())

        covered_local[s] |= incidence[:, g]
        sensor_covered[s] += n_new
        aggregate_covered += n_new

        remaining_gateways[s].remove(g)
        recompute_local_best(s)

        for req, (agg_target, sensor_targets) in req_targets.items():
            if req in selected_k:
                continue

            if aggregate_covered >= agg_target and all(
                sensor_covered[x] >= sensor_targets[x] for x in sensors
            ):
                selected_k[req] = step

        if len(selected_k) == len(requirements):
            break

    elapsed_ms = 1000.0 * (perf_counter() - t0)
    return selected_k, elapsed_ms


def exact_benchmark(
    topology: SyntheticTopology,
    requirement: tuple[int, int],
) -> tuple[dict, float]:
    """Run the sensor-separable exact DP on one synthetic instance."""
    t0 = perf_counter()
    result = solve_exact_min_links_generic(
        P_min_pct=requirement[0],
        S_min_pct=requirement[1],
        sensors=topology.sensors,
        pattern_lookup_by_sensor=topology.pattern_lookup_by_sensor,
        sensor_total=topology.sensor_total,
        total_packets=topology.total_packets,
        sensor_gateway_lists=topology.sensor_gateway_lists,
        max_budget=topology.n_links,
    )
    elapsed_ms = 1000.0 * (perf_counter() - t0)
    return result, elapsed_ms


def admissible_option_counts(
    topology: SyntheticTopology,
    S_min_pct: float,
) -> np.ndarray:
    """Count locally admissible gateway subsets for every sensor."""
    counts = []
    for s in topology.sensors:
        options = sensor_candidate_options_generic(
            sensor=s,
            S_min_pct=S_min_pct,
            pattern_lookup_s=topology.pattern_lookup_by_sensor[s],
            total_s=topology.sensor_total[s],
            gateways=topology.sensor_gateway_lists[s],
        )
        counts.append(len(options))
    return np.asarray(counts, dtype=int)
