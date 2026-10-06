"""Frozen-window sensitivity experiment for the revised manuscript."""

import argparse
from pathlib import Path

from lgc.analysis.temporal import (
    build_temporal_context,
    run_frozen_window_sensitivity,
)
from lgc.config import REQUIREMENTS, TEMPORAL_FROZEN_INITIAL_MONTHS, Paths
from lgc.io import load_dataset, save_csv, validate_analysis_dataset
from lgc.model import LinkGraph


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = Paths.resolve(args.data_root, args.out_dir)

    print(f"Loading dataset from {paths.metadata}")
    df = load_dataset(paths.metadata)
    validate_analysis_dataset(df)

    graph = LinkGraph(df)
    context = build_temporal_context(graph)

    print("[36] frozen initial-window sensitivity")
    summary, per_month = run_frozen_window_sensitivity(
        graph,
        context,
        REQUIREMENTS,
        lam=args.lam,
        frozen_initial_months=TEMPORAL_FROZEN_INITIAL_MONTHS,
    )
    save_csv(
        summary,
        paths.out_dir,
        "36_frozen_initial_window_sensitivity.csv",
    )
    save_csv(
        per_month,
        paths.out_dir,
        "36b_frozen_initial_window_sensitivity_per_month.csv",
    )

    print("\nFrozen-window sensitivity complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
