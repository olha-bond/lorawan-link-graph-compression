"""Publication-ready figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import Bbox

IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

def apply_ieee_style() -> None:
    plt.rcParams.update(IEEE_RC)

_METHOD_MARKERS = {
    "Random-k": "o",
    "Top-k reliability": "s",
    "Top-k RSSI/SNR": "^",
    "Fairness-aware greedy": "D",
}
_METHOD_LINESTYLES = {
    "Random-k": (0, (1, 1)),
    "Top-k reliability": (0, (4, 2)),
    "Top-k RSSI/SNR": (0, (3, 1, 1, 1)),
    "Fairness-aware greedy": "-",
}

def save_coverage_vs_k(
    comparison: pd.DataFrame, out_dir: Path, stem: str = "coverage_vs_k"
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    methods = [
        "Random-k",
        "Top-k reliability",
        "Top-k RSSI/SNR",
        "Fairness-aware greedy",
    ]

    for method in methods:
        sub = comparison[comparison["method"] == method].sort_values("k_links")
        if sub.empty:
            continue
        for ax_i, col in enumerate(
            ["packet_coverage_pct", "worst_sensor_coverage_pct"]
        ):
            axes[ax_i].plot(
                sub["k_links"],
                sub[col],
                marker=_METHOD_MARKERS[method],
                linestyle=_METHOD_LINESTYLES[method],
                label=method,
                markerfacecolor="white",
                markeredgewidth=0.9,
            )

    axes[0].set_xlabel("Number of selected links, k")
    axes[0].set_ylabel("Packet coverage (%)")
    axes[0].set_title("(a) Aggregate packet coverage")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Number of selected links, k")
    axes[1].set_ylabel("Worst-sensor coverage (%)")
    axes[1].set_title("(b) Worst-sensor coverage")
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"Saved: {p}")
    plt.close(fig)

_LAMBDA_STYLE = {
    10: {"marker": "o", "linestyle": "-"},
    12: {"marker": "s", "linestyle": (0, (4, 2))},
    16: {"marker": "^", "linestyle": (0, (1, 1))},
}

def save_lambda_vs_worst_coverage(
    lambda_summary: pd.DataFrame,
    ref_ks: list[int],
    out_dir: Path,
    stem: str = "lambda_vs_worst_coverage",
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for k in ref_ks:
        sub = lambda_summary[lambda_summary["k_links"] == k].sort_values("lambda")
        if sub.empty:
            continue
        style = _LAMBDA_STYLE.get(k, {"marker": "o", "linestyle": "-"})
        ax.plot(
            sub["lambda"],
            sub["worst_sensor_coverage_pct"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=f"k = {k}",
            markerfacecolor="white",
            markeredgewidth=0.9,
        )
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"Fairness weight $\lambda$ (symlog scale)")
    ax.set_ylabel("Worst-sensor coverage (%)")

    title = ax.set_title("Fairness term as a starvation-prevention trigger")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.canvas.draw()
    title.set_text("")
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    fig_w, fig_h = fig.get_size_inches()
    pad = 0.02
    crop = Bbox.from_extents(0, 0, fig_w, min(fig_h, tight.y1 + pad))

    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches=crop)
        print(f"Saved: {p}")
    plt.close(fig)

_TEMPORAL_METHOD_LABELS = {
    "proposed": "Fairness-aware",
    "multicover": "Multi-cover",
    "coverage_only": "Coverage-only",
}

def save_temporal_summary(
    summary: pd.DataFrame,
    out_dir: Path,
    stem: str = "temporal_summary",
) -> None:
    subset = summary[
        (summary["policy"] == "expanding_window")
        & (summary["evaluation_window"] == "common_excluding_new_sensor_cold_starts")
        & summary["fit_method"].isin(_TEMPORAL_METHOD_LABELS)
    ].copy()
    if subset.empty:
        return

    subset["requirement"] = subset.apply(
        lambda row: f"{int(row.P_min_pct)}/{int(row.S_min_pct)}", axis=1
    )
    requirements = ["90/80", "95/85", "98/90"]
    methods = ["proposed", "multicover", "coverage_only"]
    x = pd.RangeIndex(len(requirements)).to_numpy(dtype=float)
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for index, method in enumerate(methods):
        values = (
            subset[subset["fit_method"] == method]
            .set_index("requirement")
            .reindex(requirements)
        )
        offset = (index - 1) * width
        label = _TEMPORAL_METHOD_LABELS[method]
        axes[0].bar(
            x + offset,
            100.0 * values["pass_rate_both"],
            width,
            label=label,
        )
        axes[1].bar(
            x + offset,
            values["mean_n_links"],
            width,
            label=label,
        )

    axes[0].set_ylabel("SLA pass rate (%)")
    axes[0].set_title("(a) Prospective SLA attainment")
    axes[0].set_ylim(0, 100)
    axes[1].set_ylabel("Mean selected links")
    axes[1].set_title("(b) Link budget")
    for axis in axes:
        axis.set_xticks(x, requirements)
        axis.set_xlabel(r"Requirement $(P_{min}/S_{min})$")
        axis.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    fig.tight_layout()
    for extension in ("pdf", "png"):
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)
