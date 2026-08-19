"""Synthetic scalability experiment calibrated to the UVA link-coverage distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lgc.analysis.scalability import (
    admissible_option_counts,
    empirical_link_coverage_profiles,
    exact_benchmark,
    fairness_aware_prefix_benchmark,
    generate_synthetic_topology,
)
from lgc.config import Paths, REQUIREMENTS
from lgc.io import load_dataset, save_csv, validate_analysis_dataset


def _parse_int_list(value: str) -> list[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    try:
        return [int(x) for x in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--packets-per-sensor", type=int, default=500)
    p.add_argument("--sensor-grid", type=_parse_int_list, default=[10, 50, 100, 200, 500])
    p.add_argument("--gateway-grid", type=_parse_int_list, default=[3, 5, 10, 20])
    p.add_argument("--gateway-scaling-sensors", type=int, default=50)
    p.add_argument(
        "--exact-max-gateways",
        type=int,
        default=10,
        help="Skip exact DP above this gateway count; greedy still runs.",
    )
    p.add_argument(
        "--admissible-s-min",
        type=float,
        default=85.0,
        help="Per-sensor threshold used to summarize admissible local subsets.",
    )
    return p.parse_args()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=float), [0.1, 0.5, 0.9])
    return {"q10": float(q[0]), "median": float(q[1]), "q90": float(q[2])}


def main() -> int:
    args = _parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")

    paths = Paths.resolve(args.data_root, args.out_dir)
    df = load_dataset(paths.metadata)
    validate_analysis_dataset(df)

    profiles = empirical_link_coverage_profiles(df)
    pool = profiles["coverage_fraction"].to_numpy(dtype=float)
    save_csv(profiles, paths.out_dir, "31_empirical_link_coverage_profiles.csv")

    configurations: list[tuple[str, int, int]] = []
    for n_sensors in args.sensor_grid:
        configurations.append(("sensor_scaling", n_sensors, 3))
    for n_gateways in args.gateway_grid:
        key = ("gateway_scaling", args.gateway_scaling_sensors, n_gateways)
        if key not in configurations:
            configurations.append(key)

    rows: list[dict] = []
    calibration_rows: list[dict] = []

    for family, n_sensors, n_gateways in configurations:
        print(f"\n[{family}] S={n_sensors}, G={n_gateways}")
        for seed in range(args.seeds):
            topology = generate_synthetic_topology(
                empirical_probabilities=pool,
                n_sensors=n_sensors,
                n_gateways=n_gateways,
                packets_per_sensor=args.packets_per_sensor,
                seed=seed,
            )

            greedy_k, greedy_ms = fairness_aware_prefix_benchmark(
                topology, requirements=list(REQUIREMENTS), lam=1.0
            )

            admissible = None
            if n_gateways <= args.exact_max_gateways:
                admissible = admissible_option_counts(
                    topology, S_min_pct=args.admissible_s_min
                )

            for P_min, S_min in REQUIREMENTS:
                exact_status = "Skipped"
                exact_k = np.nan
                exact_ms = np.nan
                gap = np.nan

                if n_gateways <= args.exact_max_gateways:
                    result, exact_ms = exact_benchmark(topology, (P_min, S_min))
                    exact_status = result["status"]
                    exact_k = (
                        float(result["n_links"])
                        if result["n_links"] is not None
                        else np.nan
                    )
                    if result["n_links"] is not None and (P_min, S_min) in greedy_k:
                        gap = float(greedy_k[(P_min, S_min)] - result["n_links"])

                rows.append(
                    {
                        "family": family,
                        "n_sensors": n_sensors,
                        "n_gateways": n_gateways,
                        "n_links": topology.n_links,
                        "packets_per_sensor": args.packets_per_sensor,
                        "total_observed_packets": topology.total_packets,
                        "seed": seed,
                        "P_min_pct": P_min,
                        "S_min_pct": S_min,
                        "greedy_k": greedy_k.get((P_min, S_min), np.nan),
                        "greedy_prefix_runtime_ms": greedy_ms,
                        "exact_status": exact_status,
                        "exact_k": exact_k,
                        "exact_runtime_ms": exact_ms,
                        "greedy_gap_links": gap,
                        "local_subsets_enumerated_per_sensor": 2**n_gateways,
                        "admissible_subsets_median_at_S85": (
                            float(np.median(admissible))
                            if admissible is not None
                            else np.nan
                        ),
                        "admissible_subsets_min_at_S85": (
                            int(admissible.min()) if admissible is not None else np.nan
                        ),
                        "admissible_subsets_max_at_S85": (
                            int(admissible.max()) if admissible is not None else np.nan
                        ),
                    }
                )

            target_q = _quantiles(topology.target_link_probabilities)
            realized_q = _quantiles(topology.realized_link_probabilities)
            calibration_rows.append(
                {
                    "family": family,
                    "n_sensors": n_sensors,
                    "n_gateways": n_gateways,
                    "seed": seed,
                    "target_q10": target_q["q10"],
                    "target_median": target_q["median"],
                    "target_q90": target_q["q90"],
                    "realized_q10": realized_q["q10"],
                    "realized_median": realized_q["median"],
                    "realized_q90": realized_q["q90"],
                }
            )

    raw = pd.DataFrame(rows)
    save_csv(raw, paths.out_dir, "32_scalability_raw.csv")

    summary = (
        raw.groupby(
            ["family", "n_sensors", "n_gateways", "n_links", "P_min_pct", "S_min_pct"],
            dropna=False,
        )
        .agg(
            greedy_k_median=("greedy_k", "median"),
            greedy_runtime_median_ms=("greedy_prefix_runtime_ms", "median"),
            exact_k_median=("exact_k", "median"),
            exact_runtime_median_ms=("exact_runtime_ms", "median"),
            greedy_gap_median_links=("greedy_gap_links", "median"),
            greedy_gap_max_links=("greedy_gap_links", "max"),
            admissible_subsets_median_at_S85=(
                "admissible_subsets_median_at_S85",
                "median",
            ),
            local_subsets_enumerated_per_sensor=(
                "local_subsets_enumerated_per_sensor",
                "first",
            ),
        )
        .reset_index()
    )
    save_csv(summary, paths.out_dir, "32b_scalability_summary.csv")

    calibration = pd.DataFrame(calibration_rows)
    save_csv(calibration, paths.out_dir, "32c_scalability_calibration.csv")

    metadata = {
        "synthetic_model": (
            "Per-link conditional coverage probabilities are sampled with replacement "
            "from the 30 empirical UVA sensor-gateway coverage fractions. Independent "
            "Bernoulli reception rates are calibrated so that, after conditioning on "
            "at least one gateway reception, the target marginal coverage fractions "
            "are recovered in expectation."
        ),
        "packets_per_sensor": args.packets_per_sensor,
        "seeds": args.seeds,
        "sensor_grid": args.sensor_grid,
        "gateway_grid": args.gateway_grid,
        "gateway_scaling_sensors": args.gateway_scaling_sensors,
        "exact_max_gateways": args.exact_max_gateways,
        "reference_requirements": [list(x) for x in REQUIREMENTS],
        "empirical_probability_quantiles": _quantiles(pool),
        "important_scope_note": (
            "The synthetic generator preserves the marginal per-link coverage "
            "distribution, not the full empirical cross-gateway dependence structure."
        ),
    }
    (paths.out_dir / "32d_scalability_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("\nScalability experiment complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
