"""Stage 1: spatial analysis of the reception structure."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lgc.analysis.gateway_subsets import (
    evaluate_gateway_subsets,
    evaluate_gateway_subsets_monthly,
    summarize_monthly_gateway_subsets,
)
from lgc.analysis.link_stability import (
    build_link_temporal_stability,
    build_monthly_link_statistics,
)
from lgc.analysis.redundancy import (
    build_gateway_combination_counts,
    build_link_summary,
    build_packet_reception,
    build_redundancy_distribution,
    build_sensor_gateway_coverage,
)
from lgc.baselines.reliability import (
    build_reliability_ranking,
    reliability_ranking_from_utility,
)
from lgc.config import K_GRID, Paths
from lgc.io import (
    build_dataset_manifest,
    load_dataset,
    save_csv,
    save_json,
    validate_analysis_dataset,
)
from lgc.metrics import evaluate_link_subset
from lgc.model import LinkGraph

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Directory containing dataset/lorawan_metadata/...",
    )
    p.add_argument(
        "--out-dir", required=True, type=Path, help="Directory for CSV outputs."
    )
    p.add_argument(
        "--compact-outputs",
        action="store_true",
        help="Do not write the very large packet-level CSV (02); all analyses still run.",
    )
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = Paths.resolve(args.data_root, args.out_dir)

    print(f"Loading dataset from {paths.metadata}")
    df = load_dataset(paths.metadata)
    validate_analysis_dataset(df)
    print(f"Metadata shape: {df.shape}")
    save_json(
        build_dataset_manifest(df, paths.metadata),
        paths.out_dir,
        "00_dataset_manifest.json",
    )

    link_summary = build_link_summary(df)
    save_csv(link_summary, paths.out_dir, "01_link_summary.csv")

    packet_rx = build_packet_reception(df)
    if not args.compact_outputs:
        save_csv(packet_rx, paths.out_dir, "02_packet_reception_by_sensor_counter.csv")
    else:
        print("Skipped: 02_packet_reception_by_sensor_counter.csv (--compact-outputs)")
    save_csv(
        build_redundancy_distribution(packet_rx),
        paths.out_dir,
        "03_redundancy_distribution.csv",
    )
    save_csv(
        build_gateway_combination_counts(packet_rx),
        paths.out_dir,
        "04_gateway_combination_counts.csv",
    )

    sensor_gateway_coverage = build_sensor_gateway_coverage(df, packet_rx)
    save_csv(sensor_gateway_coverage, paths.out_dir, "05_sensor_gateway_coverage.csv")

    save_csv(
        evaluate_gateway_subsets(df, packet_rx),
        paths.out_dir,
        "06_gateway_subset_evaluation.csv",
    )
    monthly_eval = evaluate_gateway_subsets_monthly(df)
    save_csv(monthly_eval, paths.out_dir, "07_monthly_gateway_subset_evaluation.csv")
    save_csv(
        summarize_monthly_gateway_subsets(monthly_eval),
        paths.out_dir,
        "08_monthly_gateway_subset_summary.csv",
    )

    monthly_link = build_monthly_link_statistics(df)
    save_csv(monthly_link, paths.out_dir, "09_monthly_link_statistics.csv")
    link_stability = build_link_temporal_stability(monthly_link)
    save_csv(link_stability, paths.out_dir, "10_link_temporal_stability.csv")

    utility = build_reliability_ranking(
        link_summary, sensor_gateway_coverage, link_stability
    )
    utility_out = utility.sort_values("exploratory_reliability_score", ascending=False)
    if "link_id" not in utility_out.columns:
        utility_out = utility_out.assign(
            link_id=utility_out["sensor"] + "→" + utility_out["gateway"]
        )
    save_csv(utility_out, paths.out_dir, "13_exploratory_link_utility_ranking.csv")

    _write_topk_link_subset(df, utility_out, paths.out_dir, packet_rx)

    print("\nStage 1 complete.")
    return 0

def _write_topk_link_subset(
    df: pd.DataFrame, utility: pd.DataFrame, out_dir: Path, packet_rx: pd.DataFrame
) -> None:
    graph = LinkGraph(df)

    ranked_reliability = reliability_ranking_from_utility(utility)

    all_link_ids = graph.all_link_ids
    rows: list[dict] = []
    for k in K_GRID:
        k_eff = min(k, len(all_link_ids))
        rows.append(
            evaluate_link_subset(
                graph, ranked_reliability[:k_eff], f"top{k_eff}_reliability"
            )
        )

        rng = np.random.default_rng(42)
        random_batch: list[dict] = []
        for _ in range(100):
            picked = rng.choice(all_link_ids, size=k_eff, replace=False).tolist()
            random_batch.append(evaluate_link_subset(graph, picked, f"random{k_eff}"))
        rand_df = pd.DataFrame(random_batch)
        rows.append(
            {
                "selection_name": f"random{k_eff}_mean100",
                "k_links": k_eff,
                "packet_coverage_pct": rand_df["packet_coverage_pct"].mean(),
                "worst_sensor_coverage_pct": rand_df[
                    "worst_sensor_coverage_pct"
                ].mean(),
                "median_sensor_coverage_pct": rand_df[
                    "median_sensor_coverage_pct"
                ].mean(),
                "mean_sensor_coverage_pct": rand_df["mean_sensor_coverage_pct"].mean(),
                "best_rssi_median": rand_df["best_rssi_median"].mean(),
                "best_snr_median": rand_df["best_snr_median"].mean(),
            }
        )

    topk_eval = pd.DataFrame(rows).sort_values(["k_links", "selection_name"])
    save_csv(topk_eval, out_dir, "14_topk_link_subset_evaluation.csv")

if __name__ == "__main__":
    raise SystemExit(main())
