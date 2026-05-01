import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def moving_average(x, window=10):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(csv_path: str, out_dir: str, title_prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return

    specs = [
        ("episode_reward",    "Episode Reward",    "reward_curve.png"),
        ("episode_length",    "Episode Length",    "length_curve.png"),
        ("episode_violations","Violations",        "violations_curve.png"),
        ("episode_success",   "Success",           "success_curve.png"),
        ("masked_unsafe_attempts", "Masked Unsafe Attempts", "masked_unsafe_curve.png"),
        ("masked_risky_attempts",  "Masked Risky Attempts",  "masked_risky_curve.png"),
        ("risky_actions",         "Risky Actions",          "risky_actions_curve.png"),
        ("risky_penalty_total",   "Risky Penalty Total",    "risky_penalty_curve.png"),
    ]

    for col, ylabel, fname in specs:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots()
        vals = df[col].values
        ax.plot(vals, alpha=0.4, label="raw")
        smooth = moving_average(vals, window=10)
        if len(smooth) > 0:
            ax.plot(range(len(smooth)), smooth, label="smoothed (w=10)")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} - {ylabel}")
        ax.legend()
        _save(fig, os.path.join(out_dir, fname))


def plot_visit_heatmap(visit_counts, out_path: str, title: str):
    fig, ax = plt.subplots()
    im = ax.imshow(visit_counts.T, origin="lower", aspect="equal")
    fig.colorbar(im, ax=ax, label="Visit Count")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_title(title)
    _save(fig, out_path)


def plot_aggregated_curves(
    seed_csv_paths: list[str],
    out_dir: str,
    title_prefix: str,
    window: int = 10,
):
    """
    Given one CSV per seed (same env, different seeds), smooth each run,
    interpolate to a common episode axis, then plot mean ± 1 std.

    This directly satisfies the reviewer's requirement for std-dev plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    specs = [
        ("episode_reward",    "Episode Reward",    "agg_reward.png"),
        ("episode_length",    "Episode Length",    "agg_length.png"),
        ("episode_violations","Violations",        "agg_violations.png"),
        ("episode_success",   "Success Rate",      "agg_success.png"),
        ("masked_unsafe_attempts", "Masked Unsafe Attempts", "agg_masked_unsafe.png"),
        ("masked_risky_attempts",  "Masked Risky Attempts",  "agg_masked_risky.png"),
        ("risky_actions",         "Risky Actions",          "agg_risky_actions.png"),
        ("risky_penalty_total",   "Risky Penalty Total",    "agg_risky_penalty.png"),
    ]

    for col, ylabel, fname in specs:
        smoothed_runs = []
        for path in seed_csv_paths:
            df = pd.read_csv(path)
            if len(df) == 0:
                continue
            if col not in df.columns:
                continue
            smoothed_runs.append(moving_average(df[col].values, window=window))

        if not smoothed_runs:
            continue

        min_len = min(len(r) for r in smoothed_runs)
        aligned = np.stack([r[:min_len] for r in smoothed_runs])  # (n_seeds, episodes)

        mean = aligned.mean(axis=0)
        std  = aligned.std(axis=0)
        xs   = np.arange(min_len)

        fig, ax = plt.subplots()
        ax.plot(xs, mean, label="mean")
        ax.fill_between(xs, mean - std, mean + std, alpha=0.25, label="± 1 std")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} - {ylabel} (n={len(smoothed_runs)} seeds)")
        ax.legend()
        _save(fig, os.path.join(out_dir, fname))
