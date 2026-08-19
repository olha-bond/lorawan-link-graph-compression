"""Wide threshold-grid evaluation."""

import hashlib
from time import perf_counter

import pandas as pd

from ..exact.dp import solve_exact_min_links
from ..greedy.multicover import multicover_greedy_fast
from ..metrics import (
    evaluate_link_subset,
    min_k_for_requirement,
    row_at_k,
    selected_up_to,
)
from ..model import LinkGraph

def subset_hash(links_iterable) -> str:
    canonical = ",".join(sorted(links_iterable)) if links_iterable else ""
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

def run_wide_grid(
    graph: LinkGraph,
    grid_pairs: list[tuple[int, int]],
    fairness_greedy_full: pd.DataFrame,
    coverage_only_ordering: pd.DataFrame,
    progress_every: int = 100,
) -> pd.DataFrame:
    rows: list[dict] = []
    t0 = perf_counter()

    for idx, (P_min, S_min) in enumerate(grid_pairs):
        t = perf_counter()
        dp_res = solve_exact_min_links(graph, float(P_min), float(S_min))
        dt_exact = 1000.0 * (perf_counter() - t)
        if dp_res["status"] == "Optimal":
            exact_k = dp_res["n_links"]
            exact_links = sorted(dp_res["selected_links"])
            exact_hash = subset_hash(exact_links)
            ev = evaluate_link_subset(graph, dp_res["selected_links"], "exact")
            exact_pkt = float(ev["packet_coverage_pct"])
            exact_worst = float(ev["worst_sensor_coverage_pct"])
        else:
            exact_k = None
            exact_links = []
            exact_hash = ""
            exact_pkt = exact_worst = float("nan")

        rows.append(
            _grid_row(
                P_min,
                S_min,
                "Exact DP",
                exact_k,
                exact_k,
                exact_pkt,
                exact_worst,
                exact_links,
                exact_hash,
                dt_exact,
            )
        )

        t = perf_counter()
        mcov = multicover_greedy_fast(graph, float(P_min), float(S_min))
        dt_mcov = 1000.0 * (perf_counter() - t)
        rows.append(
            _grid_row_from_ordering(
                P_min,
                S_min,
                "Multi-cover greedy",
                mcov,
                exact_k,
                dt_mcov,
            )
        )

        rows.append(
            _grid_row_from_ordering(
                P_min,
                S_min,
                "Proposed fairness greedy (lambda=1)",
                fairness_greedy_full,
                exact_k,
                float("nan"),
            )
        )

        rows.append(
            _grid_row_from_ordering(
                P_min,
                S_min,
                "Coverage-only greedy (lambda=0)",
                coverage_only_ordering,
                exact_k,
                float("nan"),
            )
        )

        if (idx + 1) % progress_every == 0:
            elapsed = perf_counter() - t0
            eta = elapsed / (idx + 1) * (len(grid_pairs) - idx - 1)
            print(
                f"  progress: {idx + 1} / {len(grid_pairs)} pairs "
                f"({elapsed:.1f}s elapsed, ~{eta:.0f}s ETA)"
            )

    return pd.DataFrame(rows)

def _grid_row(
    P_min: int,
    S_min: int,
    method: str,
    k: int | None,
    exact_k: int | None,
    pkt: float,
    worst: float,
    links: list[str],
    subset_hash_str: str,
    runtime_ms: float,
) -> dict:
    return {
        "P_min_pct": P_min,
        "S_min_pct": S_min,
        "method": method,
        "n_links": k,
        "gap_to_exact": (
            (k - exact_k) if (k is not None and exact_k is not None) else None
        ),
        "packet_coverage_pct": pkt,
        "worst_sensor_coverage_pct": worst,
        "excess_packet_coverage_pp": (pkt - P_min) if k is not None else float("nan"),
        "excess_worst_sensor_coverage_pp": (worst - S_min)
        if k is not None
        else float("nan"),
        "selected_links": ";".join(links),
        "subset_hash": subset_hash_str,
        "runtime_ms": runtime_ms,
    }

def _grid_row_from_ordering(
    P_min: int,
    S_min: int,
    method: str,
    ordering: pd.DataFrame,
    exact_k: int | None,
    runtime_ms: float,
) -> dict:
    k = min_k_for_requirement(ordering, P_min, S_min)
    if k is not None:
        r = row_at_k(ordering, k)
        pkt = float(r["packet_coverage_pct"])
        worst = float(r["worst_sensor_coverage_pct"])
        links = selected_up_to(ordering, k)
        h = subset_hash(links)
    else:
        pkt = worst = float("nan")
        links = []
        h = ""
    return _grid_row(P_min, S_min, method, k, exact_k, pkt, worst, links, h, runtime_ms)

def summarize_per_method(grid_df: pd.DataFrame) -> pd.DataFrame:
    feasible = grid_df[grid_df["method"] == "Exact DP"].dropna(subset=["n_links"])
    n_feasible = len(feasible)
    rows: list[dict] = []
    for method in [
        "Multi-cover greedy",
        "Proposed fairness greedy (lambda=1)",
        "Coverage-only greedy (lambda=0)",
    ]:
        m_rows = grid_df[grid_df["method"] == method].dropna(
            subset=["n_links", "gap_to_exact"]
        )
        n_evaluated = len(m_rows)
        n_optimal = int((m_rows["gap_to_exact"] == 0).sum())
        rows.append(
            {
                "method": method,
                "n_evaluated_pairs": n_evaluated,
                "n_infeasible_pairs": n_feasible - n_evaluated,
                "optimal_cardinality_pct": (
                    100.0 * n_optimal / n_feasible if n_feasible else float("nan")
                ),
                "optimal_cardinality_pct_given_evaluated": (
                    100.0 * n_optimal / n_evaluated if n_evaluated else float("nan")
                ),
                "mean_gap": float(m_rows["gap_to_exact"].mean())
                if n_evaluated
                else float("nan"),
                "median_gap": float(m_rows["gap_to_exact"].median())
                if n_evaluated
                else float("nan"),
                "max_gap": int(m_rows["gap_to_exact"].max()) if n_evaluated else None,
            }
        )
    return pd.DataFrame(rows)

def proposed_vs_multicover_breakdown(
    grid_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prop = grid_df[
        grid_df["method"] == "Proposed fairness greedy (lambda=1)"
    ].set_index(["P_min_pct", "S_min_pct"])[
        ["n_links", "subset_hash", "selected_links"]
    ]
    mcov = grid_df[grid_df["method"] == "Multi-cover greedy"].set_index(
        ["P_min_pct", "S_min_pct"]
    )[["n_links", "subset_hash", "selected_links"]]
    exact = (
        grid_df[grid_df["method"] == "Exact DP"]
        .set_index(["P_min_pct", "S_min_pct"])[["n_links", "subset_hash"]]
        .rename(
            columns={"n_links": "n_links_exact", "subset_hash": "subset_hash_exact"}
        )
    )

    cross = prop.join(mcov, lsuffix="_prop", rsuffix="_mcov").join(exact)
    cross_valid = cross.dropna(subset=["n_links_prop", "n_links_mcov"])

    prop_better = int((cross_valid["n_links_prop"] < cross_valid["n_links_mcov"]).sum())
    mcov_better = int((cross_valid["n_links_prop"] > cross_valid["n_links_mcov"]).sum())
    equal = cross_valid[cross_valid["n_links_prop"] == cross_valid["n_links_mcov"]]
    # compare contents, not hashes
    equal_same = int((equal["selected_links_prop"] == equal["selected_links_mcov"]).sum())
    equal_diff = int((equal["selected_links_prop"] != equal["selected_links_mcov"]).sum())

    summary = pd.DataFrame(
        [
            {
                "n_pairs_where_both_feasible": len(cross_valid),
                "proposed_strictly_better_count": prop_better,
                "multicover_strictly_better_count": mcov_better,
                "equal_k_same_subset_count": equal_same,
                "equal_k_different_subset_count": equal_diff,
            }
        ]
    )

    diverge_mask = (cross_valid["n_links_prop"] != cross_valid["n_links_mcov"]) | (
        cross_valid["selected_links_prop"] != cross_valid["selected_links_mcov"]
    )
    divergences = cross_valid[diverge_mask].reset_index()

    return summary, divergences
