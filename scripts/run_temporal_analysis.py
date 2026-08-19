"""Stage 3: retrospective and rolling temporal re-optimization analyses."""

import argparse
from pathlib import Path

import pandas as pd

from lgc.analysis.temporal import (
    build_temporal_context,
    build_temporal_requirement_table,
    run_retrospective_monthly,
    run_rolling_prospective,
    summarize_retrospective_monthly,
    summarize_rolling,
)
from lgc.config import (
    LAMBDA_MAIN,
    REQUIREMENTS,
    TEMPORAL_FROZEN_INITIAL_MONTHS,
    Paths,
)
from lgc.greedy.fairness_aware import fairness_aware_greedy
from lgc.io import (
    load_dataset,
    save_csv,
    validate_analysis_dataset,
)
from lgc.model import LinkGraph
from lgc.plotting import apply_ieee_style, save_temporal_summary

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--frozen-initial-months",
        type=int,
        default=TEMPORAL_FROZEN_INITIAL_MONTHS,
        help=(
            "Number of initial months used to fit the prospective frozen-static "
            "baseline (default: %(default)s)."
        ),
    )
    return parser.parse_args(argv)

def _build_exact_vs_greedy_detail(detail: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["n_links", "meets_both_thresholds"]
    pivot = detail.pivot_table(
        index=["P_min_pct", "S_min_pct", "month"],
        columns="method",
        values=value_cols,
        aggfunc="first",
    )
    pivot.columns = [f"{value}_{method}" for value, method in pivot.columns]
    pivot = pivot.reset_index()

    for method in ["proposed", "multicover", "coverage_only"]:
        exact_col = "n_links_exact"
        method_col = f"n_links_{method}"
        pivot[f"{method}_gap_from_exact"] = pivot[method_col] - pivot[exact_col]
        pivot[f"{method}_matches_exact"] = (
            pivot[method_col].notna()
            & pivot[exact_col].notna()
            & (pivot[method_col] == pivot[exact_col])
        )
    return pivot

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = Paths.resolve(args.data_root, args.out_dir)
    apply_ieee_style()

    print(f"Loading dataset from {paths.metadata}")
    df = load_dataset(paths.metadata)
    validate_analysis_dataset(df)
    graph = LinkGraph(df)
    context = build_temporal_context(graph)
    print(graph)
    print(
        f"Temporal months: {context.months[0]} to {context.months[-1]} "
        f"({len(context.months)} calendar months)"
    )

    proposed_ordering = fairness_aware_greedy(graph, lam=LAMBDA_MAIN)

    print("[23] retrospective month-specific re-optimization")
    retrospective = run_retrospective_monthly(
        graph, context, REQUIREMENTS, lam=LAMBDA_MAIN
    )
    save_csv(
        retrospective,
        paths.out_dir,
        "23_retrospective_monthly_reoptimization.csv",
    )
    retrospective_summary = summarize_retrospective_monthly(retrospective)
    save_csv(
        retrospective_summary,
        paths.out_dir,
        "23b_retrospective_monthly_reoptimization_summary.csv",
    )

    print("[24] temporal requirement table and exact-vs-greedy detail")
    temporal_table = build_temporal_requirement_table(
        graph,
        context,
        REQUIREMENTS,
        retrospective,
        proposed_ordering,
    )
    save_csv(temporal_table, paths.out_dir, "24_temporal_requirement_table.csv")
    save_csv(
        _build_exact_vs_greedy_detail(retrospective),
        paths.out_dir,
        "24b_per_month_exact_vs_greedy.csv",
    )

    print("[30] rolling prospective temporal evaluation")
    rolling = run_rolling_prospective(
        graph,
        context,
        REQUIREMENTS,
        lam=LAMBDA_MAIN,
        frozen_initial_months=args.frozen_initial_months,
    )
    save_csv(rolling, paths.out_dir, "30_rolling_temporal_per_month.csv")
    rolling_summary = summarize_rolling(
        rolling,
        context,
        frozen_initial_months=args.frozen_initial_months,
    )
    save_csv(rolling_summary, paths.out_dir, "30b_rolling_temporal_summary.csv")
    save_temporal_summary(rolling_summary, paths.out_dir)

    print("\nStage 3 complete.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
