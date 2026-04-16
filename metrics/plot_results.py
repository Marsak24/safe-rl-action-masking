import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def moving_average(x, window=10):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_training_curves(csv_path: str, out_dir: str, title_prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return

    # Reward plot
    plt.figure()
    plt.plot(df["episode_reward"].values)
    smooth = moving_average(df["episode_reward"].values, window=10)
    if len(smooth) > 0:
        plt.plot(range(len(smooth)), smooth)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(f"{title_prefix} - Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_curve.png"))
    plt.close()

    # Length plot
    plt.figure()
    plt.plot(df["episode_length"].values)
    smooth = moving_average(df["episode_length"].values, window=10)
    if len(smooth) > 0:
        plt.plot(range(len(smooth)), smooth)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title(f"{title_prefix} - Episode Length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "length_curve.png"))
    plt.close()

    # Violations plot
    plt.figure()
    plt.plot(df["episode_violations"].values)
    smooth = moving_average(df["episode_violations"].values, window=10)
    if len(smooth) > 0:
        plt.plot(range(len(smooth)), smooth)
    plt.xlabel("Episode")
    plt.ylabel("Violations")
    plt.title(f"{title_prefix} - Safety Violations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "violations_curve.png"))
    plt.close()

    # Success plot
    plt.figure()
    plt.plot(df["episode_success"].values)
    smooth = moving_average(df["episode_success"].values, window=10)
    if len(smooth) > 0:
        plt.plot(range(len(smooth)), smooth)
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.title(f"{title_prefix} - Success")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "success_curve.png"))
    plt.close()


def plot_visit_heatmap(visit_counts, out_path: str, title: str):
    plt.figure()
    plt.imshow(visit_counts.T, origin="lower", aspect="equal")
    plt.colorbar(label="Visit Count")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
