"""Temporal link-selection analyses."""

from .context import build_temporal_context, combine_window
from .fitting import (
    evaluate_subset_on_month,
    fit_from_combined_window,
    fit_on_window,
    subset_hash,
)
from .models import CombinedWindow, FitResult, SubsetEvaluation, TemporalContext
from .retrospective import (
    run_retrospective_monthly,
    summarize_retrospective_monthly,
)
from .robust import build_temporal_requirement_table, solve_exact_all_month_static
from .rolling import run_rolling_prospective, summarize_rolling

__all__ = [
    "CombinedWindow",
    "FitResult",
    "SubsetEvaluation",
    "TemporalContext",
    "build_temporal_context",
    "build_temporal_requirement_table",
    "combine_window",
    "evaluate_subset_on_month",
    "fit_from_combined_window",
    "fit_on_window",
    "run_retrospective_monthly",
    "run_rolling_prospective",
    "solve_exact_all_month_static",
    "subset_hash",
    "summarize_retrospective_monthly",
    "summarize_rolling",
]
