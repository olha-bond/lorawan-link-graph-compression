"""Method-comparison tables for link-budget and coverage analyses."""

import pandas as pd

from ..metrics import (
    evaluate_link_subset,
    min_k_for_requirement,
    row_at_k,
    selected_up_to,
)
from ..model import LinkGraph

def _pick_at_k(df: pd.DataFrame, k: int, method_label: str) -> dict | None:
    row = df[df["k_links"] == k]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "method": method_label,
        "k_links": k,
        "packet_coverage_pct": row["packet_coverage_pct"],
        "worst_sensor_coverage_pct": row["worst_sensor_coverage_pct"],
        "median_sensor_coverage_pct": row["median_sensor_coverage_pct"],
    }

def build_method_comparison(
    k_grid: list[int],
    random_k: pd.DataFrame,
    reliability_dense: pd.DataFrame,
    rssi_snr_dense: pd.DataFrame,
    fairness_greedy: pd.DataFrame,
    n_links_max: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for k in k_grid:
        k_eff = min(k, n_links_max)
        for src, label in [
            (random_k, "Random-k"),
            (reliability_dense, "Top-k reliability"),
            (rssi_snr_dense, "Top-k RSSI/SNR"),
            (fairness_greedy, "Fairness-aware greedy"),
        ]:
            r = _pick_at_k(src, k_eff, label)
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows).sort_values(["k_links", "method"])

def build_threshold_requirements_table(
    requirements: list[tuple[int, int]],
    reliability_dense: pd.DataFrame,
    rssi_snr_dense: pd.DataFrame,
    fairness_greedy: pd.DataFrame,
    dp_results: dict[tuple[int, int], dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for P_min, S_min in requirements:
        dp_res = dp_results[(P_min, S_min)]
        rows.append(
            {
                "requirement": f">= {P_min}% packet, >= {S_min}% worst-sensor",
                "top_k_reliability_min_links": min_k_for_requirement(
                    reliability_dense, P_min, S_min
                ),
                "top_k_rssi_snr_min_links": min_k_for_requirement(
                    rssi_snr_dense, P_min, S_min
                ),
                "fairness_greedy_min_links": min_k_for_requirement(
                    fairness_greedy, P_min, S_min
                ),
                "dp_optimum_min_links": (
                    dp_res["n_links"] if dp_res["status"] == "Optimal" else None
                ),
            }
        )
    return pd.DataFrame(rows)

def build_stronger_baseline_comparison(
    graph: LinkGraph,
    requirements: list[tuple[int, int]],
    dp_results: dict[tuple[int, int], dict],
    multicover_by_req: dict[tuple[int, int], pd.DataFrame],
    coverage_only_ordering: pd.DataFrame,
    fairness_greedy: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    def add_row(
        method: str,
        P_min: int,
        S_min: int,
        k: int | None,
        pkt: float,
        worst: float,
        links: list[str],
        exact_k: int | None,
    ) -> None:
        rows.append(
            {
                "method": method,
                "P_min_pct": P_min,
                "S_min_pct": S_min,
                "n_links": k,
                "packet_coverage_pct": pkt,
                "worst_sensor_coverage_pct": worst,
                "gap_to_exact": (
                    (k - exact_k) if (k is not None and exact_k is not None) else None
                ),
                "selected_links": ";".join(links),
            }
        )

    for P_min, S_min in requirements:
        dp_res = dp_results[(P_min, S_min)]
        exact_k = dp_res["n_links"] if dp_res["status"] == "Optimal" else None
        exact_links = (
            sorted(dp_res["selected_links"]) if dp_res["status"] == "Optimal" else []
        )

        mcov = multicover_by_req[(P_min, S_min)]
        mcov_k = min_k_for_requirement(mcov, P_min, S_min)
        if mcov_k is not None:
            r = row_at_k(mcov, mcov_k)
            add_row(
                "Multi-cover greedy",
                P_min,
                S_min,
                mcov_k,
                float(r["packet_coverage_pct"]),
                float(r["worst_sensor_coverage_pct"]),
                selected_up_to(mcov, mcov_k),
                exact_k,
            )
        else:
            add_row(
                "Multi-cover greedy",
                P_min,
                S_min,
                None,
                float("nan"),
                float("nan"),
                [],
                exact_k,
            )

        cov_k = min_k_for_requirement(coverage_only_ordering, P_min, S_min)
        if cov_k is not None:
            r = row_at_k(coverage_only_ordering, cov_k)
            add_row(
                "Coverage-only greedy (lambda=0)",
                P_min,
                S_min,
                cov_k,
                float(r["packet_coverage_pct"]),
                float(r["worst_sensor_coverage_pct"]),
                selected_up_to(coverage_only_ordering, cov_k),
                exact_k,
            )
        else:
            add_row(
                "Coverage-only greedy (lambda=0)",
                P_min,
                S_min,
                None,
                float("nan"),
                float("nan"),
                [],
                exact_k,
            )

        prop_k = min_k_for_requirement(fairness_greedy, P_min, S_min)
        if prop_k is not None:
            r = row_at_k(fairness_greedy, prop_k)
            add_row(
                "Proposed fairness greedy (lambda=1)",
                P_min,
                S_min,
                prop_k,
                float(r["packet_coverage_pct"]),
                float(r["worst_sensor_coverage_pct"]),
                selected_up_to(fairness_greedy, prop_k),
                exact_k,
            )
        else:
            add_row(
                "Proposed fairness greedy (lambda=1)",
                P_min,
                S_min,
                None,
                float("nan"),
                float("nan"),
                [],
                exact_k,
            )

        if exact_k is not None:
            ev = evaluate_link_subset(graph, dp_res["selected_links"], "exact")
            add_row(
                "Exact DP",
                P_min,
                S_min,
                exact_k,
                float(ev["packet_coverage_pct"]),
                float(ev["worst_sensor_coverage_pct"]),
                exact_links,
                exact_k,
            )
        else:
            add_row(
                "Exact DP", P_min, S_min, None, float("nan"), float("nan"), [], exact_k
            )

    return pd.DataFrame(rows)
