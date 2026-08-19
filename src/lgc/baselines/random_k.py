"""Random-k control baseline."""

import numpy as np
import pandas as pd

from ..metrics import evaluate_link_subset
from ..model import LinkGraph

def evaluate_random_k(
    graph: LinkGraph, k_grid: list[int], n_repeats: int = 100, seed: int = 42
) -> pd.DataFrame:
    all_links = graph.all_link_ids
    rows: list[dict] = []

    for k in k_grid:
        k_eff = min(k, len(all_links))
        rng = np.random.default_rng(seed)
        agg: list[dict] = []
        for _ in range(n_repeats):
            picked = rng.choice(all_links, size=k_eff, replace=False).tolist()
            agg.append(evaluate_link_subset(graph, picked, name=f"random{k_eff}"))
        df_agg = pd.DataFrame(agg)
        rows.append(
            {
                "selection_name": f"random{k_eff}_mean{n_repeats}",
                "k_links": k_eff,
                "packet_coverage_pct": df_agg["packet_coverage_pct"].mean(),
                "worst_sensor_coverage_pct": df_agg["worst_sensor_coverage_pct"].mean(),
                "median_sensor_coverage_pct": df_agg[
                    "median_sensor_coverage_pct"
                ].mean(),
                "mean_sensor_coverage_pct": df_agg["mean_sensor_coverage_pct"].mean(),
                "best_rssi_median": df_agg["best_rssi_median"].mean(),
                "best_snr_median": df_agg["best_snr_median"].mean(),
            }
        )

    return pd.DataFrame(rows)
