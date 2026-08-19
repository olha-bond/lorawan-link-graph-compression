"""Export the compact manuscript-facing results and supplementary tables."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

COMMON_FILES = [
    "00_dataset_manifest.json",
]

SPATIAL_AND_REFERENCE_FILES = [
    "18_method_comparison.csv",
    "19_threshold_requirements_table.csv",
    "19b_exact_dp_runtime.csv",
    "22b_lambda_threshold_summary_at_ref_k.csv",
    "22c_lambda_subset_identity.csv",
    "25_sensor_gap_by_lambda.csv",
    "26_stronger_baseline_threshold_comparison.csv",
    "27b_grid_summary_per_method.csv",
    "27c_proposed_vs_multicover_summary.csv",
    "27d_proposed_vs_multicover_divergences.csv",
    "28_optimum_multiplicity_main.csv",
]

JOURNAL_FILES = [
    "29_equal_cardinality_subset_temporal_quality.csv",
    "29b_proposed_vs_multicover_temporal_quality_summary.csv",
    "23_retrospective_monthly_reoptimization.csv",
    "23b_retrospective_monthly_reoptimization_summary.csv",
    "24_temporal_requirement_table.csv",
    "24b_per_month_exact_vs_greedy.csv",
    "30_rolling_temporal_per_month.csv",
    "30b_rolling_temporal_summary.csv",
]

FIGURE_FILES = [
    "coverage_vs_k.pdf",
    "lambda_vs_worst_coverage.pdf",
    "temporal_summary.pdf",
]

S1_COLUMNS = [
    "test_month",
    "P_min_pct",
    "S_min_pct",
    "policy",
    "fit_method",
    "fit_months",
    "fit_status",
    "n_links",
    "packet_coverage_pct",
    "worst_sensor_coverage_pct",
    "median_sensor_coverage_pct",
    "n_active_sensors",
    "meets_packet_threshold",
    "meets_sensor_threshold",
    "meets_both_thresholds",
    "is_initial_observation_cold_start",
    "is_before_frozen_policy_available",
]

S2_COLUMNS = [
    "test_month",
    "P_min_pct",
    "S_min_pct",
    "policy",
    "fit_method",
    "fit_months",
    "fit_status",
    "n_links",
    "test_oracle_n_links",
    "budget_minus_test_oracle",
    "subset_jaccard_with_test_oracle",
    "subset_hash",
    "selected_links",
    "is_initial_observation_cold_start",
    "is_before_frozen_policy_available",
]


def _copy_required(source: Path, destination: Path, filenames: list[str]) -> None:
    missing = [name for name in filenames if not (source / name).is_file()]
    if missing:
        formatted = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(f"Required outputs are missing:\n{formatted}")

    for filename in filenames:
        shutil.copy2(source / filename, destination / filename)


def _write_supplementary_tables(source: Path, destination: Path) -> None:
    rolling_path = source / "30_rolling_temporal_per_month.csv"
    rolling = pd.read_csv(rolling_path)

    required = set(S1_COLUMNS) | set(S2_COLUMNS)
    missing_columns = sorted(required.difference(rolling.columns))
    if missing_columns:
        formatted = ", ".join(missing_columns)
        raise ValueError(f"Rolling output is missing required columns: {formatted}")

    supplementary_dir = destination / "supplementary"
    supplementary_dir.mkdir(parents=True, exist_ok=True)
    rolling.loc[:, S1_COLUMNS].to_csv(
        supplementary_dir / "Table_S1_per_month_feasibility_and_link_budget.csv",
        index=False,
    )
    rolling.loc[:, S2_COLUMNS].to_csv(
        supplementary_dir / "Table_S2_oracle_budget_and_subset_similarity.csv",
        index=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory containing generated analysis outputs.",
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        type=Path,
        help="Destination directory for the compact manuscript-facing bundle.",
    )
    parser.add_argument(
        "--profile",
        choices=["spatial", "journal"],
        default="journal",
        help="Select the spatial-only or full journal result set.",
    )
    args = parser.parse_args()

    source = args.out_dir.expanduser().resolve()
    destination = args.bundle_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    files = list(COMMON_FILES) + list(SPATIAL_AND_REFERENCE_FILES)
    if args.profile == "journal":
        files.extend(JOURNAL_FILES)

    _copy_required(source, destination, files)

    figures_dir = destination / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _copy_required(source, figures_dir, FIGURE_FILES)

    if args.profile == "journal":
        _write_supplementary_tables(source, destination)

    print(f"Exported manuscript-facing bundle to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
