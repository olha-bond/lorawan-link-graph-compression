"""Regression checks for publication-facing summaries and bundles."""

import pandas as pd

from lgc.analysis.subset_quality import _summarize_quality
from lgc.analysis.wide_grid import proposed_vs_multicover_breakdown
from scripts.export_publication_bundle import COMMON_FILES, JOURNAL_FILES


def test_subset_equality_uses_contents_not_hash_only() -> None:
    rows = [
        {
            "P_min_pct": 90,
            "S_min_pct": 80,
            "method": "Exact DP",
            "n_links": 1,
            "subset_hash": "same-hash",
            "selected_links": "s1→a",
        },
        {
            "P_min_pct": 90,
            "S_min_pct": 80,
            "method": "Proposed fairness greedy (lambda=1)",
            "n_links": 1,
            "subset_hash": "same-hash",
            "selected_links": "s1→a",
        },
        {
            "P_min_pct": 90,
            "S_min_pct": 80,
            "method": "Multi-cover greedy",
            "n_links": 1,
            "subset_hash": "same-hash",
            "selected_links": "s1→b",
        },
    ]

    summary, divergences = proposed_vs_multicover_breakdown(pd.DataFrame(rows))

    assert summary.iloc[0]["equal_k_same_subset_count"] == 0
    assert summary.iloc[0]["equal_k_different_subset_count"] == 1
    assert len(divergences) == 1


def test_subset_quality_summary_counts_each_measure() -> None:
    quality = pd.DataFrame(
        [
            {
                "P_min_pct": 90,
                "S_min_pct": 80,
                "method": "Proposed fairness greedy (lambda=1)",
                "months_meeting_both_thresholds": 5,
                "worst_sensor_slack_pp": -2.0,
                "worst_packet_slack_pp": 1.0,
            },
            {
                "P_min_pct": 90,
                "S_min_pct": 80,
                "method": "Multi-cover greedy",
                "months_meeting_both_thresholds": 4,
                "worst_sensor_slack_pp": -2.0,
                "worst_packet_slack_pp": -1.0,
            },
        ]
    )

    summary = _summarize_quality(quality, tol=0.1).iloc[0]
    assert summary["n_pairs"] == 1
    assert summary["practical_proposed_wins"] == 1
    assert summary["d_months_meeting_both_positive_count"] == 1
    assert summary["d_worst_packet_slack_pp_positive_count"] == 1


def test_journal_bundle_contains_manifest_and_subset_quality_summary() -> None:
    assert "00_dataset_manifest.json" in COMMON_FILES
    assert (
        "29b_proposed_vs_multicover_temporal_quality_summary.csv"
        in JOURNAL_FILES
    )
