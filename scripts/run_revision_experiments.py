"""Peer-review sensitivity experiments for the revised manuscript."""

import argparse
from pathlib import Path

from lgc.analysis.temporal import (
    build_temporal_context,
    run_boundary_month_sensitivity,
)
from lgc.config import REQUIREMENTS, Paths
from lgc.io import load_dataset, save_csv, validate_analysis_dataset
from lgc.model import LinkGraph


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = Paths.resolve(args.data_root, args.out_dir)

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

    print("[33] boundary-month sensitivity")
    sensitivity = run_boundary_month_sensitivity(
        graph,
        context,
        REQUIREMENTS,
    )
    save_csv(
        sensitivity,
        paths.out_dir,
        "33_boundary_month_sensitivity.csv",
    )

    print("\nRevision experiments complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
