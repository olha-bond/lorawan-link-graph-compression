"""Heterogeneous-traffic sensitivity experiment for the revised manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lgc.analysis.scalability import (
    empirical_link_coverage_profiles,
    exact_benchmark,
    fairness_aware_prefix_benchmark,
)
from lgc.analysis.traffic_heterogeneity import (
    empirical_sensor_traffic_profile,
    generate_synthetic_topology_with_packet_counts,
    synthetic_sensor_packet_counts,
)
from lgc.config import Paths, REQUIREMENTS
from lgc.io import load_dataset, save_csv, validate_analysis_dataset


REVISION_LAMBDAS = [0.0, 0.01, 0.02, 0.1, 1.0]


def _parse_int_list(value: str) -> list[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    try:
        return [int(x) for x in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def _parse_float_list(value: str) -> list[float]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one number")
    try:
        return [float(x) for x in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--packets-per-sensor", type=int, default=500)
    p.add_argument("--sensor-grid", type=_parse_int_list, default=[10, 50, 100])
    p.add_argument("--n-gateways", type=int, default=3)
    p.add_argument("--lambdas", type=_parse_float_list, default=REVISION_LAMBDAS)
    return p.parse_args(argv)


def _traffic_stats(packet_counts: np.ndarray) -> dict[str, float]:
    counts = np.asarray(packet_counts, dtype=float)
    mean = float(counts.mean())
    return {
        "traffic_min_packets": int(counts.min()),
        "traffic_median_packets": float(np.median(counts)),
        "traffic_max_packets": int(counts.max()),
        "traffic_cv": float(counts.std(ddof=0) / mean),
        "traffic_max_min_ratio": float(counts.max() / counts.min()),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.packets_per_sensor <= 0:
        raise ValueError("--packets-per-sensor must be positive")
    if args.n_gateways <= 0:
        raise ValueError("--n-gateways must be positive")

    paths = Paths.resolve(args.data_root, args.out_dir)
    print(f"Loading dataset from {paths.metadata}")
    df = load_dataset(paths.metadata)
    validate_analysis_dataset(df)

    link_profiles = empirical_link_coverage_profiles(df)
    probability_pool = link_profiles["coverage_fraction"].to_numpy(dtype=float)

    traffic_profile = empirical_sensor_traffic_profile(df)
    traffic_pool = traffic_profile["traffic_multiplier"].to_numpy(dtype=float)

    rows: list[dict] = []

    for n_sensors in args.sensor_grid:
        for scenario, heterogeneous in [
            ("balanced", False),
            ("heterogeneous", True),
        ]:
            print(f"\n[{scenario}] S={n_sensors}, G={args.n_gateways}")
            for seed in range(args.seeds):
                packet_counts = synthetic_sensor_packet_counts(
                    empirical_multipliers=traffic_pool,
                    n_sensors=n_sensors,
                    packets_per_sensor=args.packets_per_sensor,
                    seed=seed,
                    heterogeneous=heterogeneous,
                )
                topology = generate_synthetic_topology_with_packet_counts(
                    empirical_probabilities=probability_pool,
                    packet_counts=packet_counts,
                    n_gateways=args.n_gateways,
                    seed=seed,
                )

                exact_by_req: dict[tuple[int, int], tuple[dict, float]] = {}
                for requirement in REQUIREMENTS:
                    exact_by_req[requirement] = exact_benchmark(
                        topology,
                        requirement,
                    )

                stats = _traffic_stats(packet_counts)
                for lam in args.lambdas:
                    greedy_k, greedy_ms = fairness_aware_prefix_benchmark(
                        topology,
                        requirements=list(REQUIREMENTS),
                        lam=lam,
                    )

                    for P_min, S_min in REQUIREMENTS:
                        result, exact_ms = exact_by_req[(P_min, S_min)]
                        exact_k = (
                            float(result["n_links"])
                            if result["n_links"] is not None
                            else np.nan
                        )
                        greedy_value = greedy_k.get((P_min, S_min), np.nan)
                        gap = (
                            float(greedy_value - exact_k)
                            if np.isfinite(greedy_value) and np.isfinite(exact_k)
                            else np.nan
                        )

                        rows.append(
                            {
                                "traffic_scenario": scenario,
                                "n_sensors": n_sensors,
                                "n_gateways": args.n_gateways,
                                "n_links": topology.n_links,
                                "seed": seed,
                                "lambda": lam,
                                "P_min_pct": P_min,
                                "S_min_pct": S_min,
                                "total_observed_packets": topology.total_packets,
                                **stats,
                                "greedy_k": greedy_value,
                                "greedy_prefix_runtime_ms": greedy_ms,
                                "exact_status": result["status"],
                                "exact_k": exact_k,
                                "exact_runtime_ms": exact_ms,
                                "greedy_gap_links": gap,
                            }
                        )

    raw = pd.DataFrame(rows)
    save_csv(raw, paths.out_dir, "34_heterogeneous_traffic_lambda_raw.csv")

    summary = (
        raw.groupby(
            [
                "traffic_scenario",
                "n_sensors",
                "n_gateways",
                "P_min_pct",
                "S_min_pct",
                "lambda",
            ],
            dropna=False,
        )
        .agg(
            runs=("seed", "nunique"),
            greedy_k_median=("greedy_k", "median"),
            exact_k_median=("exact_k", "median"),
            greedy_gap_mean_links=("greedy_gap_links", "mean"),
            greedy_gap_median_links=("greedy_gap_links", "median"),
            greedy_gap_max_links=("greedy_gap_links", "max"),
            optimum_match_rate=(
                "greedy_gap_links",
                lambda x: float((x.dropna() == 0).mean()) if x.notna().any() else np.nan,
            ),
            greedy_runtime_median_ms=("greedy_prefix_runtime_ms", "median"),
            traffic_cv_median=("traffic_cv", "median"),
            traffic_max_min_ratio_median=("traffic_max_min_ratio", "median"),
        )
        .reset_index()
    )
    save_csv(
        summary,
        paths.out_dir,
        "34b_heterogeneous_traffic_lambda_summary.csv",
    )

    metadata = {
        "traffic_model": (
            "Balanced scenarios assign the same observed-packet count to every sensor. "
            "Heterogeneous scenarios preserve the same total observed-packet count but "
            "allocate it across sensors according to the empirical UVA per-sensor traffic "
            "multipliers. The empirical multiplier profile is repeated and shuffled for "
            "larger synthetic deployments."
        ),
        "paired_topology_note": (
            "For a fixed seed and sensor index, balanced and heterogeneous scenarios use "
            "the same sampled target link-coverage profile; only the per-sensor packet "
            "count changes."
        ),
        "packets_per_sensor_reference": args.packets_per_sensor,
        "seeds": args.seeds,
        "sensor_grid": args.sensor_grid,
        "n_gateways": args.n_gateways,
        "lambdas": args.lambdas,
        "reference_requirements": [list(x) for x in REQUIREMENTS],
        "empirical_sensor_traffic_profile": traffic_profile.to_dict(orient="records"),
    }
    (paths.out_dir / "34c_heterogeneous_traffic_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nHeterogeneous-traffic sensitivity complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
