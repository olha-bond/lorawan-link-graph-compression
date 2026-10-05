"""Heterogeneous synthetic-traffic sensitivity helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scalability import (
    SyntheticTopology,
    _generate_observed_incidence,
    _pattern_lookup_from_incidence,
    _sample_conditional_profile,
)


def empirical_sensor_traffic_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return observed-packet counts and relative traffic volume per sensor."""
    packets = df[["sensor", "counter"]].drop_duplicates()
    result = (
        packets.groupby("sensor")
        .size()
        .rename("observed_packets")
        .reset_index()
        .sort_values("sensor")
        .reset_index(drop=True)
    )
    mean_packets = float(result["observed_packets"].mean())
    result["traffic_multiplier"] = result["observed_packets"] / mean_packets
    return result


def _allocate_packet_counts(weights: np.ndarray, total_packets: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0 or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("traffic weights must be finite and positive")
    if total_packets < len(weights):
        raise ValueError("total packet count must be at least the number of sensors")

    counts = np.ones(len(weights), dtype=int)
    remaining = total_packets - len(weights)
    if remaining == 0:
        return counts

    raw = remaining * weights / weights.sum()
    extra = np.floor(raw).astype(int)
    delta = int(remaining - extra.sum())
    if delta > 0:
        order = np.argsort(-(raw - extra), kind="stable")
        extra[order[:delta]] += 1

    return counts + extra


def synthetic_sensor_packet_counts(
    empirical_multipliers: np.ndarray,
    n_sensors: int,
    packets_per_sensor: int,
    seed: int,
    heterogeneous: bool,
) -> np.ndarray:
    """Build balanced or empirically heterogeneous sensor packet counts."""
    if n_sensors <= 0 or packets_per_sensor <= 0:
        raise ValueError("sensor and packet counts must be positive")

    if not heterogeneous:
        return np.full(n_sensors, packets_per_sensor, dtype=int)

    pool = np.asarray(empirical_multipliers, dtype=float)
    pool = pool[np.isfinite(pool) & (pool > 0)]
    if len(pool) == 0:
        raise ValueError("empirical traffic multiplier pool is empty")

    repeats = int(np.ceil(n_sensors / len(pool)))
    weights = np.tile(pool, repeats)[:n_sensors].copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(weights)

    return _allocate_packet_counts(
        weights,
        total_packets=n_sensors * packets_per_sensor,
    )


def generate_synthetic_topology_with_packet_counts(
    empirical_probabilities: np.ndarray,
    packet_counts: np.ndarray,
    n_gateways: int,
    seed: int,
) -> SyntheticTopology:
    """Generate a synthetic topology with per-sensor observed-packet counts."""
    counts = np.asarray(packet_counts, dtype=int)
    if len(counts) == 0 or n_gateways <= 0 or np.any(counts <= 0):
        raise ValueError("topology dimensions and packet counts must be positive")

    pool = np.asarray(empirical_probabilities, dtype=float)
    pool = pool[np.isfinite(pool)]
    if len(pool) == 0:
        raise ValueError("empirical probability pool is empty")

    gateway_names = [f"g{j + 1:02d}" for j in range(n_gateways)]
    incidence_by_sensor: dict[str, np.ndarray] = {}
    pattern_lookup_by_sensor: dict[str, list[tuple[frozenset[str], int]]] = {}
    sensor_total: dict[str, int] = {}
    sensor_gateway_lists: dict[str, list[str]] = {}
    targets: list[np.ndarray] = []
    realized: list[np.ndarray] = []

    for i, n_packets in enumerate(counts.tolist()):
        sensor = f"s{i + 1:04d}"
        profile_seed, incidence_seed = np.random.SeedSequence([seed, i]).spawn(2)
        profile_rng = np.random.default_rng(profile_seed)
        incidence_rng = np.random.default_rng(incidence_seed)

        target = _sample_conditional_profile(pool, n_gateways, profile_rng)
        incidence = _generate_observed_incidence(
            target,
            n_packets,
            incidence_rng,
        )

        incidence_by_sensor[sensor] = incidence
        pattern_lookup_by_sensor[sensor] = _pattern_lookup_from_incidence(
            incidence,
            gateway_names,
        )
        sensor_total[sensor] = n_packets
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
