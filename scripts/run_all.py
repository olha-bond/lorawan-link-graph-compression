"""Run the spatial and link-selection pipeline, optionally including temporal analysis."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_link_selection
import run_spatial_analysis
import run_temporal_analysis

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--skip-wide-grid", action="store_true")
    parser.add_argument("--skip-subset-quality", action="store_true")
    parser.add_argument(
        "--compact-outputs",
        action="store_true",
        help="Skip the very large packet-level output 02 CSV.",
    )
    parser.add_argument(
        "--include-temporal",
        action="store_true",
        help="Also run retrospective and rolling temporal analyses.",
    )
    parser.add_argument(
        "--frozen-initial-months",
        type=int,
        default=3,
        help="Initial training-window length for the frozen prospective baseline.",
    )
    args = parser.parse_args()

    stage1_argv = ["--data-root", str(args.data_root), "--out-dir", str(args.out_dir)]
    if args.compact_outputs:
        stage1_argv.append("--compact-outputs")
    rc = run_spatial_analysis.main(stage1_argv)
    if rc != 0:
        return rc

    stage2_argv = ["--data-root", str(args.data_root), "--out-dir", str(args.out_dir)]
    if args.skip_wide_grid:
        stage2_argv.append("--skip-wide-grid")
    if args.skip_subset_quality:
        stage2_argv.append("--skip-subset-quality")
    rc = run_link_selection.main(stage2_argv)
    if rc != 0:
        return rc

    if args.include_temporal:
        stage3_argv = [
            "--data-root",
            str(args.data_root),
            "--out-dir",
            str(args.out_dir),
            "--frozen-initial-months",
            str(args.frozen_initial_months),
        ]
        rc = run_temporal_analysis.main(stage3_argv)
        if rc != 0:
            return rc

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
