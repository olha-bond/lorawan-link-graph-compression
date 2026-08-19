"""Month-level quality of divergent equal-cardinality link subsets."""

from collections import defaultdict
from time import perf_counter

import numpy as np
import pandas as pd

from ..model import LinkGraph

def _evaluate_monthly_fast(
    graph: LinkGraph,
    subset_links: list[str],
    sm_idx: np.ndarray,
    sm_totals: np.ndarray,
    sm_decode: list[tuple[str, str]],
    months_sorted: list[str],
) -> pd.DataFrame:
    subset_words = np.zeros(graph.packet_link_words.shape[1], dtype=np.uint32)
    for link in subset_links:
        column = graph.link_to_col[link]
        word_index, bit_index = divmod(column, 32)
        subset_words[word_index] |= np.uint32(1 << bit_index)
    covered = (graph.packet_link_words & subset_words).any(axis=1)
    sm_covered = np.bincount(sm_idx[covered], minlength=len(sm_decode))

    per_month_cov: dict[str, int] = defaultdict(int)
    per_month_tot: dict[str, int] = defaultdict(int)
    per_month_pcts: dict[str, list[float]] = defaultdict(list)
    for c, (_sensor, month) in enumerate(sm_decode):
        tot = int(sm_totals[c])
        if tot == 0:
            continue
        cov = int(sm_covered[c])
        per_month_tot[month] += tot
        per_month_cov[month] += cov
        per_month_pcts[month].append(100.0 * cov / tot)

    rows: list[dict] = []
    for m in months_sorted:
        tot = per_month_tot[m]
        cov = per_month_cov[m]
        pkt_pct = 100.0 * cov / tot if tot > 0 else float("nan")
        pcts = per_month_pcts[m]
        worst = min(pcts) if pcts else float("nan")
        rows.append(
            {
                "month": m,
                "packet_coverage_pct": pkt_pct,
                "worst_sensor_coverage_pct": worst,
            }
        )
    return pd.DataFrame(rows)

def _secondary_metrics(monthly_df: pd.DataFrame, P_min: int, S_min: int) -> dict:
    pkt = monthly_df["packet_coverage_pct"].dropna()
    ws = monthly_df["worst_sensor_coverage_pct"].dropna()
    both_valid = monthly_df.dropna(
        subset=["packet_coverage_pct", "worst_sensor_coverage_pct"]
    )

    min_pkt = float(pkt.min()) if len(pkt) else float("nan")
    min_ws = float(ws.min()) if len(ws) else float("nan")
    mean_pkt = float(pkt.mean()) if len(pkt) else float("nan")
    mean_ws = float(ws.mean()) if len(ws) else float("nan")

    pkt_shortfall = (
        float(np.maximum(0, P_min - pkt).mean()) if len(pkt) else float("nan")
    )
    ws_shortfall = float(np.maximum(0, S_min - ws).mean()) if len(ws) else float("nan")

    months_pass_pkt = int((pkt >= P_min).sum())
    months_pass_ws = int((ws >= S_min).sum())
    months_pass_both = int(
        (
            (both_valid["packet_coverage_pct"] >= P_min)
            & (both_valid["worst_sensor_coverage_pct"] >= S_min)
        ).sum()
    )

    return {
        "min_month_packet_coverage_pct": min_pkt,
        "min_month_worst_sensor_coverage_pct": min_ws,
        "worst_packet_slack_pp": min_pkt - P_min
        if not np.isnan(min_pkt)
        else float("nan"),
        "worst_sensor_slack_pp": min_ws - S_min
        if not np.isnan(min_ws)
        else float("nan"),
        "months_meeting_packet_threshold": months_pass_pkt,
        "months_meeting_sensor_threshold": months_pass_ws,
        "months_meeting_both_thresholds": months_pass_both,
        "mean_packet_shortfall_pp": pkt_shortfall,
        "mean_sensor_shortfall_pp": ws_shortfall,
        "mean_month_packet_coverage_pct": mean_pkt,
        "mean_month_worst_sensor_coverage_pct": mean_ws,
    }

def run_subset_quality(
    graph: LinkGraph, divergences: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensor_month_pairs = list(
        zip(graph.all_packets["sensor"], graph.all_packets["month"])
    )
    unique_sm = sorted(set(sensor_month_pairs))
    sm_to_code = {sm: c for c, sm in enumerate(unique_sm)}
    sm_idx = np.array([sm_to_code[sm] for sm in sensor_month_pairs], dtype=np.int64)
    sm_totals = np.bincount(sm_idx, minlength=len(unique_sm))
    months_sorted = sorted({m for _, m in unique_sm})

    monthly_cache: dict[str, pd.DataFrame] = {}

    def get_monthly(subset_links_str: str) -> pd.DataFrame:
        # cache by full contents, not hash
        if subset_links_str not in monthly_cache:
            links = subset_links_str.split(";") if subset_links_str else []
            monthly_cache[subset_links_str] = _evaluate_monthly_fast(
                graph, links, sm_idx, sm_totals, unique_sm, months_sorted
            )
        return monthly_cache[subset_links_str]

    quality_rows: list[dict] = []
    t0 = perf_counter()
    for _, row in divergences.iterrows():
        P_min = int(row["P_min_pct"])
        S_min = int(row["S_min_pct"])
        for method, hash_col, links_col, k_col in [
            (
                "Proposed fairness greedy (lambda=1)",
                "subset_hash_prop",
                "selected_links_prop",
                "n_links_prop",
            ),
            (
                "Multi-cover greedy",
                "subset_hash_mcov",
                "selected_links_mcov",
                "n_links_mcov",
            ),
        ]:
            h = row[hash_col]
            links_str = row[links_col]
            monthly = get_monthly(links_str)
            metrics = _secondary_metrics(monthly, P_min, S_min)
            quality_rows.append(
                {
                    "P_min_pct": P_min,
                    "S_min_pct": S_min,
                    "method": method,
                    "n_links": int(row[k_col]),
                    "selected_links": links_str,
                    "subset_hash": h,
                    **metrics,
                }
            )
    elapsed = perf_counter() - t0
    print(
        f"  processed {len(divergences)} divergent pairs, "
        f"cached {len(monthly_cache)} unique subsets, {elapsed:.1f}s elapsed"
    )

    quality_df = pd.DataFrame(quality_rows)
    summary_df = _summarize_quality(quality_df)
    return quality_df, summary_df

def _summarize_quality(quality_df: pd.DataFrame, tol: float = 0.1) -> pd.DataFrame:
    if quality_df.empty:
        return pd.DataFrame([{"n_pairs": 0}])

    prop = quality_df[
        quality_df["method"] == "Proposed fairness greedy (lambda=1)"
    ].set_index(["P_min_pct", "S_min_pct"])
    mcov = quality_df[quality_df["method"] == "Multi-cover greedy"].set_index(
        ["P_min_pct", "S_min_pct"]
    )

    def compare(prop_m: dict, mcov_m: dict, t: float) -> str:
        a, b = (
            prop_m["months_meeting_both_thresholds"],
            mcov_m["months_meeting_both_thresholds"],
        )
        if a != b:
            return "proposed" if a > b else "multicover"
        a, b = prop_m["worst_sensor_slack_pp"], mcov_m["worst_sensor_slack_pp"]
        if not (np.isnan(a) or np.isnan(b)) and abs(a - b) > t:
            return "proposed" if a > b else "multicover"
        a, b = prop_m["worst_packet_slack_pp"], mcov_m["worst_packet_slack_pp"]
        if not (np.isnan(a) or np.isnan(b)) and abs(a - b) > t:
            return "proposed" if a > b else "multicover"
        return "tie"

    winners_practical: list[str] = []
    winners_strict: list[str] = []
    d_sensor_slack: list[float] = []
    d_packet_slack: list[float] = []
    d_months_both: list[float] = []

    for pair in prop.index:
        prop_m = prop.loc[pair].to_dict()
        mcov_m = mcov.loc[pair].to_dict()
        winners_practical.append(compare(prop_m, mcov_m, tol))
        winners_strict.append(compare(prop_m, mcov_m, 0.0))
        d_sensor_slack.append(
            prop_m["worst_sensor_slack_pp"] - mcov_m["worst_sensor_slack_pp"]
        )
        d_packet_slack.append(
            prop_m["worst_packet_slack_pp"] - mcov_m["worst_packet_slack_pp"]
        )
        d_months_both.append(
            prop_m["months_meeting_both_thresholds"]
            - mcov_m["months_meeting_both_thresholds"]
        )

    def counts(winners: list[str]) -> dict:
        return {
            "proposed_wins": sum(1 for w in winners if w == "proposed"),
            "multicover_wins": sum(1 for w in winners if w == "multicover"),
            "ties": sum(1 for w in winners if w == "tie"),
        }

    def distribution(values: list[float], name: str) -> dict:
        arr = np.array(
            [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        )
        if len(arr) == 0:
            return {}
        return {
            f"{name}_median": float(np.median(arr)),
            f"{name}_positive_count": int((arr > tol).sum()),
            f"{name}_zero_count": int((np.abs(arr) <= tol).sum()),
            f"{name}_negative_count": int((arr < -tol).sum()),
        }

    summary = {
        "n_pairs": len(prop),
        "tie_tolerance_pp": tol,
        **{f"practical_{k}": v for k, v in counts(winners_practical).items()},
        **{f"strict_{k}": v for k, v in counts(winners_strict).items()},
        **distribution(d_sensor_slack, "d_worst_sensor_slack_pp"),
        **distribution(d_packet_slack, "d_worst_packet_slack_pp"),
        **distribution(d_months_both, "d_months_meeting_both"),
    }
    return pd.DataFrame([summary])
