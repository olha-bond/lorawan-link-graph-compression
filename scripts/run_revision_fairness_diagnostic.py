"""Fairness tie diagnostic for the revised manuscript."""

import argparse
from pathlib import Path

from lgc.analysis.fairness_diagnostic import fairness_worst_tie_diagnostic
from lgc.io import save_csv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[35] fairness worst-tie diagnostic")
    diagnostic = fairness_worst_tie_diagnostic(lam=args.lam)
    save_csv(
        diagnostic,
        args.out_dir,
        "35_fairness_worst_tie_diagnostic.csv",
    )

    print("\nFairness diagnostic complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
