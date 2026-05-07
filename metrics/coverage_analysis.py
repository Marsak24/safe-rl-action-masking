"""
metrics/coverage_analysis.py
=============================
Computes state-coverage metrics from visit_counts.npy files produced
during training, and saves a summary CSV + all plots to
results/coverage_analysis/.

Metrics computed per method × env × seed
-----------------------------------------
  coverage_rate   : unique cells visited / navigable cells
                    (navigable = union of all cells ever reached by any
                     method across all seeds for that env — best proxy
                     without re-instantiating the environment)
  visit_entropy   : Shannon entropy of the visit distribution
                    H = -sum_s p(s) log p(s), p(s) = visits(s)/total
                    Higher → more spread-out exploration
  peak_fraction   : max_cell_visits / total_visits
                    Lower → less path repetition / looping

Plots produced
--------------
  results/coverage_analysis/
    coverage_metrics.csv          — per-seed and mean/std rows
    heatmaps/
      <env>_<method>_mean_heatmap.png   — mean visit map averaged over seeds
      <env>_all_methods.png             — side-by-side comparison per env
    charts/
      <env>_coverage_rate.png
      <env>_visit_entropy.png
      <env>_peak_fraction.png
      all_envs_overview.png             — 3×3 subplot overview

Run
---
    python -m metrics.coverage_analysis
"""

from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Configuration — edit if result dirs ever change
# ---------------------------------------------------------------------------

BASE = Path("results")

# label → (result_base_dir, env_tag_suffix)
# env_tag_suffix is the suffix used in the folder name, e.g. "lavagaps5"
METHODS: dict[str, str] = {
    "Vanilla PPO":     "vanilla_ppo",
    "Hard Masking":    "masked_ppo",
    "Soft Masking":    "soft_action_masked_ppo",
    "Hybrid (p=0.1)":  "soft_masked_ppo",
    "Hybrid (p=0.01)": "soft_masked_ppo_p001",
}

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

SEEDS = [0, 1, 2, 3, 4]

OUT_DIR      = Path("results/coverage_analysis")
HEATMAP_DIR  = OUT_DIR / "heatmaps"
CHART_DIR    = OUT_DIR / "charts"

# Colour palette — one colour per method (order matches METHODS)
PALETTE = ["#555555", "#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_visit_counts(method_dir: str, env_tag: str, seed: int) -> np.ndarray | None:
    """Load visit_counts.npy for a given method / env / seed, return None if missing."""
    path = BASE / method_dir / env_tag / f"seed_{seed}" / "visit_counts.npy"
    if not path.exists():
        return None
    vc = np.load(path).astype(np.float64)
    return vc


def _coverage_rate(vc: np.ndarray, navigable_mask: np.ndarray) -> float:
    """Fraction of navigable cells visited at least once."""
    navigable_n = int(navigable_mask.sum())
    if navigable_n == 0:
        return float("nan")
    visited = int((vc[navigable_mask] > 0).sum())
    return visited / navigable_n


def _visit_entropy(vc: np.ndarray) -> float:
    """Shannon entropy of the visit distribution (base-2 bits)."""
    flat = vc.flatten()
    total = flat.sum()
    if total == 0:
        return 0.0
    probs = flat[flat > 0] / total
    return float(scipy_entropy(probs, base=2))


def _peak_fraction(vc: np.ndarray) -> float:
    """Fraction of all visits concentrated in the single most-visited cell."""
    total = vc.sum()
    if total == 0:
        return float("nan")
    return float(vc.max() / total)


def _navigable_mask(env_tag: str) -> np.ndarray:
    """
    Build a navigable-cell mask for an env as the union of all cells ever
    visited by any method across all seeds.  This avoids reimplementing
    MiniGrid's grid parser while remaining accurate for reachable cells.
    """
    union: np.ndarray | None = None
    for method_dir in METHODS.values():
        for seed in SEEDS:
            vc = _load_visit_counts(method_dir, env_tag, seed)
            if vc is None:
                continue
            if union is None:
                union = (vc > 0)
            else:
                union = union | (vc > 0)
    if union is None:
        raise RuntimeError(f"No visit_counts.npy found for env {env_tag}")
    return union


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect_metrics() -> list[dict]:
    """
    Returns a list of dicts, one per (method, env, seed), with keys:
        method, env_id, seed,
        coverage_rate, visit_entropy, peak_fraction,
        total_visits, unique_cells, navigable_cells
    """
    rows: list[dict] = []

    for env_id in ENV_IDS:
        env_tag = ENV_TAG[env_id]
        print(f"\n[coverage] {env_id}")

        try:
            nav_mask = _navigable_mask(env_tag)
        except RuntimeError as e:
            print(f"  SKIP: {e}")
            continue

        navigable_n = int(nav_mask.sum())

        for label, method_dir in METHODS.items():
            for seed in SEEDS:
                vc = _load_visit_counts(method_dir, env_tag, seed)
                if vc is None:
                    continue

                cr  = _coverage_rate(vc, nav_mask)
                ent = _visit_entropy(vc)
                pf  = _peak_fraction(vc)
                tot = int(vc.sum())
                uniq = int((vc[nav_mask] > 0).sum())

                rows.append(dict(
                    method          = label,
                    env_id          = env_id,
                    seed            = seed,
                    coverage_rate   = cr,
                    visit_entropy   = ent,
                    peak_fraction   = pf,
                    total_visits    = tot,
                    unique_cells    = uniq,
                    navigable_cells = navigable_n,
                ))
                print(f"  {label:20s} seed={seed}  cov={cr:.3f}  H={ent:.2f} bits  peak={pf:.3f}")

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Also write an aggregated (mean ± std over seeds) version
    agg_rows: list[dict] = []
    metric_cols = ["coverage_rate", "visit_entropy", "peak_fraction",
                   "total_visits", "unique_cells"]
    seen: set[tuple] = set()
    for r in rows:
        key = (r["method"], r["env_id"])
        if key in seen:
            continue
        seen.add(key)
        group = [x for x in rows if x["method"] == r["method"] and x["env_id"] == r["env_id"]]
        agg = dict(method=r["method"], env_id=r["env_id"], n_seeds=len(group),
                   navigable_cells=r["navigable_cells"])
        for col in metric_cols:
            vals = [x[col] for x in group]
            agg[f"{col}_mean"] = float(np.mean(vals))
            agg[f"{col}_std"]  = float(np.std(vals))
        agg_rows.append(agg)

    agg_path = path.parent / (path.stem + "_aggregated.csv")
    agg_fields = list(agg_rows[0].keys()) if agg_rows else []
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"\n[coverage] Saved {path}")
    print(f"[coverage] Saved {agg_path}")


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_mean_heatmap(
    method_dir: str, label: str, env_id: str, env_tag: str,
    nav_mask: np.ndarray,
) -> np.ndarray | None:
    """Average visit_counts across seeds and return the mean array (also saves png)."""
    arrays = []
    for seed in SEEDS:
        vc = _load_visit_counts(method_dir, env_tag, seed)
        if vc is not None:
            arrays.append(vc)
    if not arrays:
        return None

    mean_vc = np.mean(np.stack(arrays, axis=0), axis=0)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mean_vc.T, origin="lower", aspect="equal", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Mean visits")
    # Overlay navigable boundary in grey
    ax.contour(nav_mask.T.astype(float), levels=[0.5], colors="steelblue",
               linewidths=0.8, linestyles="--", alpha=0.6)
    ax.set_title(f"{label}\n{env_id}", fontsize=9)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    _save(fig, HEATMAP_DIR / f"{env_tag}_{safe_label}_mean_heatmap.png")
    return mean_vc


def plot_all_methods_heatmap(env_id: str, env_tag: str, nav_mask: np.ndarray) -> None:
    """Side-by-side heatmap for all methods, one figure per env."""
    method_items = [(label, d) for label, d in METHODS.items()]
    n = len(method_items)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.8))
    fig.suptitle(f"Mean Visit Heatmaps — {env_id}", fontsize=12, y=1.02)

    all_means = []
    for label, method_dir in method_items:
        arrays = []
        for seed in SEEDS:
            vc = _load_visit_counts(method_dir, env_tag, seed)
            if vc is not None:
                arrays.append(vc)
        if arrays:
            all_means.append(np.mean(np.stack(arrays, axis=0), axis=0))
        else:
            all_means.append(None)

    # Shared colour scale
    vmax_global = max((m.max() for m in all_means if m is not None), default=1.0)

    for ax, (label, _), mean_vc in zip(axes, method_items, all_means):
        if mean_vc is None:
            ax.set_title(label + "\n(no data)", fontsize=8)
            ax.axis("off")
            continue
        norm = mcolors.LogNorm(vmin=max(mean_vc[mean_vc > 0].min(), 1e-1),
                               vmax=vmax_global) if mean_vc.max() > 0 else None
        im = ax.imshow(mean_vc.T, origin="lower", aspect="equal",
                       cmap="YlOrRd", norm=norm)
        ax.contour(nav_mask.T.astype(float), levels=[0.5], colors="steelblue",
                   linewidths=0.8, linestyles="--", alpha=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Grid X", fontsize=7)
        ax.set_ylabel("Grid Y", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, shrink=0.75)

    _save(fig, HEATMAP_DIR / f"{env_tag}_all_methods.png")
    print(f"[coverage] Heatmap grid saved: {HEATMAP_DIR / f'{env_tag}_all_methods.png'}")


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------

def _bar_chart(
    rows: list[dict],
    env_id: str,
    metric: str,
    ylabel: str,
    title: str,
    fname: str,
) -> None:
    method_labels = list(METHODS.keys())
    colours = PALETTE[: len(method_labels)]

    means, stds = [], []
    for label in method_labels:
        group = [r[metric] for r in rows
                 if r["method"] == label and r["env_id"] == env_id]
        if group:
            means.append(float(np.mean(group)))
            stds.append(float(np.std(group)))
        else:
            means.append(float("nan"))
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(method_labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colours, alpha=0.85,
                  error_kw=dict(elinewidth=1.2, ecolor="black"))

    # Value annotations on top of bars
    for bar, m, s in zip(bars, means, stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — {env_id}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, CHART_DIR / fname)


def plot_bar_charts(rows: list[dict]) -> None:
    specs = [
        ("coverage_rate",  "Coverage Rate (fraction)", "State Coverage Rate",  "coverage_rate"),
        ("visit_entropy",  "Entropy (bits)",           "Visit Distribution Entropy", "visit_entropy"),
        ("peak_fraction",  "Peak Cell Fraction",       "Peak Cell Concentration",    "peak_fraction"),
    ]
    for env_id in ENV_IDS:
        env_tag = ENV_TAG[env_id]
        for metric, ylabel, title, fname_stem in specs:
            _bar_chart(
                rows, env_id, metric, ylabel, title,
                fname=f"{env_tag}_{fname_stem}.png",
            )
    print(f"[coverage] Bar charts saved to {CHART_DIR}/")


# ---------------------------------------------------------------------------
# Overview grid (3 envs × 3 metrics)
# ---------------------------------------------------------------------------

def plot_overview_grid(rows: list[dict]) -> None:
    metrics   = ["coverage_rate", "visit_entropy", "peak_fraction"]
    ylabels   = ["Coverage Rate", "Entropy (bits)", "Peak Cell Fraction"]
    method_labels = list(METHODS.keys())
    colours   = PALETTE[: len(method_labels)]

    fig, axes = plt.subplots(len(metrics), len(ENV_IDS),
                             figsize=(5 * len(ENV_IDS), 4 * len(metrics)),
                             sharey="row")
    fig.suptitle("State Coverage Metrics — All Methods × All Environments",
                 fontsize=13, y=1.01)

    for row_idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col_idx, env_id in enumerate(ENV_IDS):
            ax = axes[row_idx][col_idx]
            means, stds = [], []
            for label in method_labels:
                group = [r[metric] for r in rows
                         if r["method"] == label and r["env_id"] == env_id]
                if group:
                    means.append(float(np.mean(group)))
                    stds.append(float(np.std(group)))
                else:
                    means.append(float("nan"))
                    stds.append(0.0)

            x = np.arange(len(method_labels))
            ax.bar(x, means, yerr=stds, capsize=3, color=colours,
                   alpha=0.85, error_kw=dict(elinewidth=1.0))
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=7)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_idx == 0:
                ax.set_title(env_id.replace("MiniGrid-", "").replace("-v0", ""),
                             fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

    _save(fig, CHART_DIR / "all_envs_overview.png")
    print(f"[coverage] Overview grid saved: {CHART_DIR / 'all_envs_overview.png'}")


# ---------------------------------------------------------------------------
# Per-seed heatmap strip (shows variance across seeds for one method)
# ---------------------------------------------------------------------------

def plot_per_seed_strip(method_dir: str, label: str, env_tag: str, env_id: str) -> None:
    """One row of heatmaps, one per seed, so variance is visible."""
    arrays = []
    for seed in SEEDS:
        vc = _load_visit_counts(method_dir, env_tag, seed)
        arrays.append(vc)

    valid = [(s, a) for s, a in zip(SEEDS, arrays) if a is not None]
    if not valid:
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]

    all_vals = np.concatenate([a.flatten() for _, a in valid])
    vmax_g = all_vals.max() or 1.0

    for ax, (seed, vc) in zip(axes, valid):
        norm = mcolors.LogNorm(vmin=max(vc[vc > 0].min(), 1e-1), vmax=vmax_g) \
               if vc.max() > 0 else None
        im = ax.imshow(vc.T, origin="lower", aspect="equal",
                       cmap="YlOrRd", norm=norm)
        ax.set_title(f"seed={seed}", fontsize=8)
        ax.set_xlabel("Grid X", fontsize=7)
        ax.set_ylabel("Grid Y", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, shrink=0.75)

    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    fig.suptitle(f"{label} — {env_id} — per-seed visits", fontsize=10)
    _save(fig, HEATMAP_DIR / "per_seed" / f"{env_tag}_{safe_label}_per_seed.png")
    print(f"[coverage] Per-seed strip saved: {env_tag}/{safe_label}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    for d in (OUT_DIR, HEATMAP_DIR, CHART_DIR, HEATMAP_DIR / "per_seed"):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. Collect metrics ------------------------------------------------
    rows = collect_metrics()
    if not rows:
        print("[coverage] No visit_counts.npy files found. Exiting.")
        return

    save_csv(rows, OUT_DIR / "coverage_metrics.csv")

    # --- 2. Heatmaps -------------------------------------------------------
    for env_id in ENV_IDS:
        env_tag = ENV_TAG[env_id]
        try:
            nav_mask = _navigable_mask(env_tag)
        except RuntimeError:
            continue

        # Individual mean heatmaps
        for label, method_dir in METHODS.items():
            plot_mean_heatmap(method_dir, label, env_id, env_tag, nav_mask)

        # Side-by-side comparison grid
        plot_all_methods_heatmap(env_id, env_tag, nav_mask)

        # Per-seed strips (variance visualisation)
        for label, method_dir in METHODS.items():
            plot_per_seed_strip(method_dir, label, env_tag, env_id)

    # --- 3. Bar charts -----------------------------------------------------
    plot_bar_charts(rows)

    # --- 4. Overview grid --------------------------------------------------
    plot_overview_grid(rows)

    print(f"\n[coverage] All outputs saved under: {OUT_DIR.resolve()}")
    print("[coverage] Done.")


if __name__ == "__main__":
    main()
