"""Stage 2: link selection, exact reference, ablations, and figures."""

import argparse
from pathlib import Path

import pandas as pd

from lgc.analysis.ablation import (
    build_sensor_gap_table,
    check_subset_identity,
    coarse_ablation,
    fine_ablation,
    monthly_stability_of_chosen_subset,
    summarize_at_ref_k,
)
from lgc.analysis.method_comparison import (
    build_method_comparison,
    build_stronger_baseline_comparison,
    build_threshold_requirements_table,
)
from lgc.analysis.multiplicity import build_multiplicity_table
from lgc.analysis.subset_quality import run_subset_quality
from lgc.analysis.wide_grid import (
    proposed_vs_multicover_breakdown,
    run_wide_grid,
    summarize_per_method,
)
from lgc.baselines.random_k import evaluate_random_k
from lgc.baselines.reliability import (
    reliability_ranking_from_utility,
    top_k_reliability_dense,
)
from lgc.baselines.rssi_snr import top_k_rssi_snr_dense
from lgc.config import (
    COARSE_LAMBDAS,
    FINE_LAMBDAS,
    GRID_P_MIN,
    GRID_S_MIN,
    K_GRID,
    LAMBDA_MAIN,
    REF_KS,
    REQUIREMENTS,
    STABILITY_REQUIREMENT,
    Paths,
)
from lgc.exact.dp import benchmark_dp, solve_exact_min_links
from lgc.greedy.fairness_aware import fairness_aware_greedy
from lgc.greedy.multicover import multicover_greedy_fast
from lgc.io import (
    load_dataset,
    save_csv,
    validate_analysis_dataset,
)
from lgc.metrics import evaluate_link_subset
from lgc.model import LinkGraph
from lgc.plotting import (
    apply_ieee_style,
    save_coverage_vs_k,
    save_lambda_vs_worst_coverage,
)

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--skip-wide-grid",
        action="store_true",
        help="Skip the 1000-pair diagnostic (slow, ~ minutes).",
    )
    p.add_argument(
        "--skip-subset-quality",
        action="store_true",
        help="Skip the retrospective subset-quality check.",
    )
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = Paths.resolve(args.data_root, args.out_dir)

    utility_path = paths.out_dir / "13_exploratory_link_utility_ranking.csv"
    if not utility_path.exists():
        raise FileNotFoundError(
            f"{utility_path} not found. Run run_spatial_analysis.py first."
        )

    apply_ieee_style()
    print(f"Loading dataset from {paths.metadata}")
    df = load_dataset(paths.metadata)
    validate_analysis_dataset(df)
    graph = LinkGraph(df)
    print(graph)

    print("[15] fairness-aware greedy")
    fairness_greedy_df = fairness_aware_greedy(graph, lam=LAMBDA_MAIN)
    save_csv(fairness_greedy_df, paths.out_dir, "15_fairness_aware_greedy_results.csv")

    link_summary = (
        df.groupby(["sensor", "gateway"])
        .agg(rssi_mean=("rssi", "mean"), snr_mean=("snr", "mean"))
        .reset_index()
    )
    rssi_snr_dense = top_k_rssi_snr_dense(graph, link_summary)
    save_csv(
        rssi_snr_dense[rssi_snr_dense["k_links"].isin(K_GRID)],
        paths.out_dir,
        "16_topk_rssi_snr_baseline.csv",
    )

    utility = pd.read_csv(utility_path)
    ranked_reliability = reliability_ranking_from_utility(utility)
    reliability_dense = top_k_reliability_dense(graph, ranked_reliability)
    save_csv(reliability_dense, paths.out_dir, "16b_topk_reliability_dense.csv")

    random_k_baseline = evaluate_random_k(graph, K_GRID, n_repeats=100, seed=42)
    random_k_baseline["k_links"] = random_k_baseline["k_links"].astype(int)
    save_csv(random_k_baseline, paths.out_dir, "17_random_k_baseline.csv")

    reliability_grid = reliability_dense[
        reliability_dense["k_links"].isin(K_GRID)
    ].copy()
    rssi_snr_grid = rssi_snr_dense[rssi_snr_dense["k_links"].isin(K_GRID)].copy()
    method_comparison = build_method_comparison(
        K_GRID,
        random_k_baseline,
        reliability_grid,
        rssi_snr_grid,
        fairness_greedy_df,
        n_links_max=graph.n_links,
    )
    save_csv(method_comparison, paths.out_dir, "18_method_comparison.csv")
    save_coverage_vs_k(method_comparison, paths.out_dir)

    dp_results: dict[tuple[int, int], dict] = {}
    print("[DP] exact optima at reference requirements")
    for P_min, S_min in REQUIREMENTS:
        res = solve_exact_min_links(graph, P_min, S_min)
        dp_results[(P_min, S_min)] = res
        if res["status"] == "Optimal":
            ev = evaluate_link_subset(graph, res["selected_links"], "exact")
            print(
                f"  ({P_min}%, {S_min}%): {res['n_links']} links "
                f"(P={ev['packet_coverage_pct']:.2f}%, "
                f"S_min={ev['worst_sensor_coverage_pct']:.2f}%)"
            )

    threshold_table = build_threshold_requirements_table(
        REQUIREMENTS,
        reliability_dense,
        rssi_snr_dense,
        fairness_greedy_df,
        dp_results,
    )
    save_csv(threshold_table, paths.out_dir, "19_threshold_requirements_table.csv")

    runtime_df = benchmark_dp(graph, REQUIREMENTS)
    save_csv(runtime_df, paths.out_dir, "19b_exact_dp_runtime.csv")

    ablation_df = coarse_ablation(graph, COARSE_LAMBDAS)
    save_csv(ablation_df, paths.out_dir, "20_ablation_lambda.csv")

    monthly_stab, chosen_k = monthly_stability_of_chosen_subset(
        graph,
        fairness_greedy_df,
        STABILITY_REQUIREMENT,
    )
    print(f"  monthly stability at k={chosen_k} for {STABILITY_REQUIREMENT}")
    save_csv(monthly_stab, paths.out_dir, "21_monthly_selected_subset_stability.csv")

    fine_df = fine_ablation(graph, FINE_LAMBDAS)
    save_csv(fine_df, paths.out_dir, "22_lambda_threshold_ablation.csv")
    lambda_summary = summarize_at_ref_k(fine_df, FINE_LAMBDAS, REF_KS)
    save_csv(lambda_summary, paths.out_dir, "22b_lambda_threshold_summary_at_ref_k.csv")
    subset_identity = check_subset_identity(fine_df, FINE_LAMBDAS, REF_KS)
    save_csv(subset_identity, paths.out_dir, "22c_lambda_subset_identity.csv")
    save_csv(
        build_sensor_gap_table(fine_df, FINE_LAMBDAS, REF_KS),
        paths.out_dir,
        "25_sensor_gap_by_lambda.csv",
    )
    save_lambda_vs_worst_coverage(lambda_summary, REF_KS, paths.out_dir)

    coverage_only_ordering = fairness_aware_greedy(graph, lam=0.0)
    multicover_by_req = {
        (P, S): multicover_greedy_fast(graph, float(P), float(S))
        for P, S in REQUIREMENTS
    }
    stronger = build_stronger_baseline_comparison(
        graph,
        REQUIREMENTS,
        dp_results,
        multicover_by_req,
        coverage_only_ordering,
        fairness_greedy_df,
    )
    save_csv(stronger, paths.out_dir, "26_stronger_baseline_threshold_comparison.csv")

    if not args.skip_wide_grid:
        grid_pairs = [(P, S) for P in GRID_P_MIN for S in GRID_S_MIN]
        print(f"Wide grid: {len(grid_pairs)} threshold pairs")
        grid_df = run_wide_grid(
            graph,
            grid_pairs,
            fairness_greedy_df,
            coverage_only_ordering,
        )
        save_csv(grid_df, paths.out_dir, "27_wide_grid_all_methods.csv")
        save_csv(
            summarize_per_method(grid_df),
            paths.out_dir,
            "27b_grid_summary_per_method.csv",
        )
        cross_summary, divergences = proposed_vs_multicover_breakdown(grid_df)
        save_csv(cross_summary, paths.out_dir, "27c_proposed_vs_multicover_summary.csv")
        save_csv(
            divergences, paths.out_dir, "27d_proposed_vs_multicover_divergences.csv"
        )

        save_csv(
            build_multiplicity_table(graph, REQUIREMENTS, dp_results),
            paths.out_dir,
            "28_optimum_multiplicity_main.csv",
        )

        if not args.skip_subset_quality and not divergences.empty:
            quality, quality_summary = run_subset_quality(graph, divergences)
            save_csv(
                quality,
                paths.out_dir,
                "29_equal_cardinality_subset_temporal_quality.csv",
            )
            save_csv(
                quality_summary,
                paths.out_dir,
                "29b_proposed_vs_multicover_temporal_quality_summary.csv",
            )

    print("\nStage 2 complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
