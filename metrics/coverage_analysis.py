"""
metrics/coverage_analysis.py
=============================
Produces meaningful comparison plots from already-collected experiment data.

Reads from:
  results/<method>/summary.csv                          (all methods)
  results/<method>/<env>/seed_*/visit_counts.npy        (methods with visit-count data)

Outputs to results/coverage_analysis/:
  eval_success_rate.png      -- success rate +/- std, all methods x 3 envs
  eval_reward.png            -- mean reward +/- std
  convergence_timestep.png   -- convergence speed (NaN -> 300k = "no convergence")
  training_violations.png    -- mean training violations +/- std
  visit_entropy.png          -- Shannon entropy of visit distribution
  overview.png               -- combined 2x2 figure
  heatmaps/
    lavagaps{5,6,7}_methods.png -- mean visit heatmap grid per env

Run from project root:
    python -m metrics.coverage_analysis
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 300_000
OUT_DIR = Path("results/coverage_analysis")

# (display label, result base dir, visit_counts base dir or None)
METHODS = [
    ("Vanilla PPO", "results/vanilla_ppo", "results/vanilla_ppo"),
    ("Penalty PPO", "results/penalty_ppo_0.5", "results/penalty_ppo_0.5"),
    ("Hard Masking", "results/masked_ppo", "results/masked_ppo"),
    ("Soft Masking", "results/soft_action_masked_ppo", "results/soft_action_masked_ppo"),
    ("Hybrid p=0.1", "results/soft_masked_ppo", "results/soft_masked_ppo"),
    ("Hybrid p=0.01", "results/soft_masked_ppo_p001", "results/soft_masked_ppo_p001"),
]

ENV_IDS = [
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",
]

ENV_TAG = {
    "MiniGrid-LavaGapS5-v0": "lavagaps5",
    "MiniGrid-LavaGapS6-v0": "lavagaps6",
    "MiniGrid-LavaGapS7-v0": "lavagaps7",
}

ENV_SHORT = {
    "MiniGrid-LavaGapS5-v0": "LavaGap S5",
    "MiniGrid-LavaGapS6-v0": "LavaGap S6",
    "MiniGrid-LavaGapS7-v0": "LavaGap S7",
}

# One colour per method, in the same order as METHODS
# COLOURS = ["#555555", "#E67E22", "#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
COLOURS = [
    "#555555",  # Vanilla PPO - dark gray
    "#D62728",  # Penalty PPO - red
    "#1F77B4",  # Hard Masking - blue
    "#FFB000",  # Soft Masking - yellow/orange
    "#2CA02C",  # Hybrid p=0.1 - green
    "#9467BD",  # Hybrid p=0.01 - purple
]
METHOD_TO_COLOUR = {
    label: COLOURS[i]
    for i, (label, _, _) in enumerate(METHODS)
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_summary(result_dir: str) -> pd.DataFrame | None:
    p = Path(result_dir) / "summary.csv"

    if not p.exists():
        print(f"  warning: missing summary file: {p}")
        return None

    return pd.read_csv(p)


def agg(
    df: pd.DataFrame,
    env_id: str,
    col: str,
    fill_nan: float | None = None,
) -> tuple[float, float]:
    if col not in df.columns:
        print(f"  warning: missing column '{col}' in summary.csv")
        return float("nan"), 0.0

    sub = df[df["env_id"] == env_id][col].copy()

    if fill_nan is not None:
        sub = sub.fillna(fill_nan)

    sub = sub.dropna()

    if len(sub) == 0:
        return float("nan"), 0.0

    return float(sub.mean()), float(sub.std())


def visit_entropy(vc_base: str, env_tag: str) -> tuple[float, float]:
    vals = []

    for seed in SEEDS:
        p = Path(vc_base) / env_tag / f"seed_{seed}" / "visit_counts.npy"

        if not p.exists():
            continue

        vc = np.load(p).astype(np.float64).flatten()
        total = vc.sum()

        if total == 0:
            continue

        prob = vc[vc > 0] / total
        entropy = float((-prob * np.log2(prob)).sum())
        vals.append(entropy)

    if not vals:
        return float("nan"), 0.0

    return float(np.mean(vals)), float(np.std(vals))


def mean_visit_grid(vc_base: str, env_tag: str) -> np.ndarray | None:
    arrays = []

    for seed in SEEDS:
        p = Path(vc_base) / env_tag / f"seed_{seed}" / "visit_counts.npy"

        if p.exists():
            arrays.append(np.load(p).astype(np.float64))

    if not arrays:
        return None

    return np.mean(np.stack(arrays), axis=0)


def remove_empty_methods(
    data: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, float]]]:
    """
    Remove methods where all means are NaN.

    This prevents invisible empty bars from being included in the legend.
    """
    cleaned = {}

    for label, values in data.items():
        means = [mean for mean, _ in values]

        if not all(np.isnan(mean) for mean in means):
            cleaned[label] = values
        else:
            print(f"  warning: no valid values found for {label}; skipping from plot")

    return cleaned


def colours_for(data: dict[str, list[tuple[float, float]]]) -> list[str]:
    """
    Return colours in the exact same order as the plotted method labels.

    This is the important fix. It avoids using slices like COLOURS[2:],
    which caused Penalty PPO to be mismatched or dropped in the entropy plot.
    """
    return [METHOD_TO_COLOUR[label] for label in data.keys()]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def _draw_grouped_bar(
    ax: plt.Axes,
    data: dict[str, list[tuple[float, float]]],
    ylabel: str,
    title: str,
    colours: list[str],
    hline: float | None = None,
    hline_label: str = "",
    ylim: tuple[float, float] | None = None,
    legend: bool = True,
) -> None:
    """Draw a grouped bar chart onto an existing Axes."""

    method_labels = list(data.keys())
    env_shorts = [ENV_SHORT[e] for e in ENV_IDS]

    n_methods = len(method_labels)
    n_envs = len(env_shorts)

    if n_methods == 0:
        ax.set_title(title, fontsize=10)
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        return

    width = 0.8 / n_methods
    x = np.arange(n_envs)

    for i, (label, colour) in enumerate(zip(method_labels, colours)):
        means = [data[label][j][0] for j in range(n_envs)]
        stds = [data[label][j][1] for j in range(n_envs)]

        offset = (i - n_methods / 2 + 0.5) * width

        ax.bar(
            x + offset,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=3,
            color=colour,
            alpha=0.85,
            label=label,
            error_kw=dict(elinewidth=1.1, ecolor="black"),
        )

    if hline is not None:
        ax.axhline(
            hline,
            color="grey",
            linestyle="--",
            linewidth=0.9,
            label=hline_label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(env_shorts, fontsize=9)
    ax.set_xlabel("Environment", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)

    if ylim:
        ax.set_ylim(*ylim)

    if legend:
        ax.legend(fontsize=7, framealpha=0.75)

    ax.grid(axis="y", linestyle="--", alpha=0.35)


def grouped_bar(
    data: dict[str, list[tuple[float, float]]],
    ylabel: str,
    title: str,
    out_path: Path,
    colours: list[str],
    hline: float | None = None,
    hline_label: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Standalone grouped bar figure saved to disk."""

    fig, ax = plt.subplots(figsize=(8, 4.5))

    _draw_grouped_bar(
        ax,
        data,
        ylabel,
        title,
        colours,
        hline=hline,
        hline_label=hline_label,
        ylim=ylim,
    )

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_eval_metrics() -> None:
    success = {}
    reward = {}
    conv = {}
    viols = {}

    for label, result_dir, _ in METHODS:
        df = load_summary(result_dir)

        if df is None:
            continue

        success[label] = [agg(df, e, "eval_success_rate") for e in ENV_IDS]
        reward[label] = [agg(df, e, "eval_mean_reward") for e in ENV_IDS]
        conv[label] = [
            agg(df, e, "convergence_timestep", fill_nan=TOTAL_TIMESTEPS)
            for e in ENV_IDS
        ]
        viols[label] = [agg(df, e, "mean_violations") for e in ENV_IDS]

    success = remove_empty_methods(success)
    reward = remove_empty_methods(reward)
    conv = remove_empty_methods(conv)
    viols = remove_empty_methods(viols)

    grouped_bar(
        success,
        "Success Rate",
        "Eval Success Rate (mean ± std over 5 seeds)",
        OUT_DIR / "eval_success_rate.png",
        colours_for(success),
        ylim=(0, 1.15),
    )

    grouped_bar(
        reward,
        "Mean Reward",
        "Eval Mean Reward (mean ± std over 5 seeds)",
        OUT_DIR / "eval_reward.png",
        colours_for(reward),
    )

    grouped_bar(
        conv,
        "Timestep",
        f"Convergence Timestep [NaN capped at {TOTAL_TIMESTEPS:,} = did not converge]\n"
        "(mean ± std over 5 seeds)",
        OUT_DIR / "convergence_timestep.png",
        colours_for(conv),
        hline=TOTAL_TIMESTEPS,
        hline_label="300k cap (no convergence)",
    )

    grouped_bar(
        viols,
        "Mean Violations / Episode",
        "Training Violations (mean ± std over 5 seeds)",
        OUT_DIR / "training_violations.png",
        colours_for(viols),
    )


def plot_visit_entropy() -> None:
    """
    Plot visit entropy for methods that have visit_counts.npy data.

    Penalty PPO is included here only if its visit_counts.npy files exist.
    """
    entropy_data = {}

    for label, _, vc_base in METHODS:
        if vc_base is None:
            continue

        entropy_data[label] = [
            visit_entropy(vc_base, ENV_TAG[e])
            for e in ENV_IDS
        ]

    entropy_data = remove_empty_methods(entropy_data)

    grouped_bar(
        entropy_data,
        "Shannon Entropy (bits)",
        "Visit Distribution Entropy [higher = more spread exploration]\n"
        "(mean ± std over 5 seeds, methods with visit-count data)",
        OUT_DIR / "visit_entropy.png",
        colours_for(entropy_data),
    )


def plot_overview() -> None:
    """Single 2x2 figure combining the 4 key metric panels."""

    success = {}
    reward = {}
    viols = {}

    for label, result_dir, _ in METHODS:
        df = load_summary(result_dir)

        if df is None:
            continue

        success[label] = [agg(df, e, "eval_success_rate") for e in ENV_IDS]
        reward[label] = [agg(df, e, "eval_mean_reward") for e in ENV_IDS]
        viols[label] = [agg(df, e, "mean_violations") for e in ENV_IDS]

    entropy_data = {}

    for label, _, vc_base in METHODS:
        if vc_base is None:
            continue

        entropy_data[label] = [
            visit_entropy(vc_base, ENV_TAG[e])
            for e in ENV_IDS
        ]

    success = remove_empty_methods(success)
    reward = remove_empty_methods(reward)
    viols = remove_empty_methods(viols)
    entropy_data = remove_empty_methods(entropy_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    fig.suptitle(
        "Safe RL Method Comparison — Key Metrics",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    _draw_grouped_bar(
        axes[0, 0],
        success,
        "Success Rate",
        "(a) Eval Success Rate",
        colours_for(success),
        ylim=(0, 1.18),
        legend=True,
    )

    _draw_grouped_bar(
        axes[0, 1],
        reward,
        "Mean Reward",
        "(b) Eval Mean Reward",
        colours_for(reward),
        legend=False,
    )

    _draw_grouped_bar(
        axes[1, 0],
        viols,
        "Mean Violations / Episode",
        "(c) Training Violations",
        colours_for(viols),
        legend=False,
    )

    _draw_grouped_bar(
        axes[1, 1],
        entropy_data,
        "Shannon Entropy (bits)",
        "(d) Visit Entropy [methods with visit-count data]",
        colours_for(entropy_data),
        legend=True,
    )

    # Shared legend for all methods, based on panel (a)
    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.04),
    )

    # Remove internal legends to avoid duplication
    for ax in axes.flat:
        legend = ax.get_legend()

        if legend:
            legend.remove()

    _save(fig, OUT_DIR / "overview.png")


def plot_heatmaps() -> None:
    """
    Plot mean visit heatmaps for all methods that have visit_counts.npy data.

    Vanilla PPO is skipped because its visit-count base directory is None.
    Penalty PPO is included if the corresponding visit_counts.npy files exist.
    """
    methods_with_visits = [
        (label, vc_base)
        for label, _, vc_base in METHODS
        if vc_base is not None
    ]

    heatmap_dir = OUT_DIR / "heatmaps"

    for env_id in ENV_IDS:
        env_tag = ENV_TAG[env_id]

        grids = [
            (label, mean_visit_grid(vc_base, env_tag))
            for label, vc_base in methods_with_visits
        ]

        grids = [
            (label, grid)
            for label, grid in grids
            if grid is not None
        ]

        if not grids:
            print(f"  warning: no visit heatmap data found for {env_tag}")
            continue

        n = len(grids)

        fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.0))

        if n == 1:
            axes = [axes]

        fig.suptitle(
            f"Mean Visit Heatmap (training) — {env_id}",
            fontsize=11,
            y=1.02,
        )

        nonzero_values = [
            grid[grid > 0].flatten()
            for _, grid in grids
            if np.any(grid > 0)
        ]

        if nonzero_values:
            nonzero_all = np.concatenate(nonzero_values)
            vmin_g = float(nonzero_all.min())
            vmax_g = float(nonzero_all.max())
        else:
            vmin_g = 1.0
            vmax_g = 1.0

        for ax, (label, mean_vc) in zip(axes, grids):
            norm = mcolors.LogNorm(
                vmin=max(vmin_g, 1e-1),
                vmax=max(vmax_g, max(vmin_g, 1e-1)),
            )

            im = ax.imshow(
                mean_vc.T,
                origin="lower",
                aspect="equal",
                cmap="YlOrRd",
                norm=norm,
            )

            plt.colorbar(im, ax=ax, shrink=0.80, label="visits (log)")

            ax.set_title(label, fontsize=9)
            ax.set_xlabel("Grid X", fontsize=8)
            ax.set_ylabel("Grid Y", fontsize=8)
            ax.tick_params(labelsize=7)

        _save(fig, heatmap_dir / f"{env_tag}_methods.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "heatmaps").mkdir(exist_ok=True)

    print("\n[1/4] Eval metric bar charts ...")
    plot_eval_metrics()

    print("\n[2/4] Visit entropy bar chart ...")
    plot_visit_entropy()

    print("\n[3/4] Combined overview figure ...")
    plot_overview()

    print("\n[4/4] Visit heatmaps ...")
    plot_heatmaps()

    print(f"\nDone. All outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()