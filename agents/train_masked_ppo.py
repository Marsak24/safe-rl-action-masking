"""
agents/train_masked_ppo.py
==========================
Method 3: Hard action masking with MaskablePPO (sb3-contrib).

The masking logic lives entirely in src/masking.py and
env/lava_masking_wrapper.py.  This script owns the training loop and
keeps the same hyperparameters, evaluation pipeline, and output layout
as the existing vanilla-PPO and penalty-PPO scripts so that results are
directly comparable.

Key differences from vanilla/penalty scripts
---------------------------------------------
* Uses MaskablePPO (sb3-contrib) instead of PPO.
* Wraps the environment with LavaMaskingWrapper + ActionMasker so the
  policy always receives a valid action mask at every step.
* Logs two extra columns: masked_unsafe_attempts, masked_risky_attempts.
* Model prediction during evaluation passes the action mask explicitly.

Directory structure produced
-----------------------------
results/masked_ppo/
    models/
        lavagaps5_seed0_masked_ppo.zip  ...
    videos/
        lavagaps5/seed_0/  ...
    lavagaps5/
        seed_0/
            training_log.csv
            visit_heatmap.png
            visit_counts.npy
            ...
        aggregated/
            ...
    summary.csv
    summary_aggregated.csv
"""

from __future__ import annotations

import os
import json
import random

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from minigrid.wrappers import FlatObsWrapper

from env.lava_masking_wrapper import LavaMaskingWrapper, make_masked_env, make_masked_video_env
from metrics.masked_training_logger import MaskedEpisodeCSVLogger
from metrics.plot_results import (
    plot_training_curves,
    plot_visit_heatmap,
    plot_aggregated_curves,
)

# ---------------------------------------------------------------------------
# Configuration (keep in sync with other method scripts)
# ---------------------------------------------------------------------------
BASE_DIR    = "results/masked_ppo"
MODELS_DIR  = os.path.join(BASE_DIR, "models")
VIDEOS_DIR  = os.path.join(BASE_DIR, "videos")
SUMMARY_CSV = os.path.join(BASE_DIR, "summary.csv")
AGG_CSV     = os.path.join(BASE_DIR, "summary_aggregated.csv")

ENV_IDS = [
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",
]

SEEDS           = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 300_000
MAX_EVAL_STEPS  = 300
N_EVAL_EPISODES = 20

# Identical to vanilla-PPO so comparisons are fair.
PPO_KWARGS = dict(
    policy        = "MlpPolicy",
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    learning_rate = 3e-4,
    gamma         = 0.99,
    ent_coef      = 0.01,
    verbose       = 1,
    device        = "cpu",
)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_train_env(env_id: str, seed: int, mask_risky: bool = False) -> gym.Env:
    """
    Training env with MaskablePPO-compatible action masking.

    Stack: ActionMasker → LavaMaskingWrapper → FlatObsWrapper → MiniGrid
    """
    return make_masked_env(env_id, seed=seed, mask_risky=mask_risky)


def make_video_env(
    env_id: str,
    video_folder: str,
    seed: int,
    mask_risky: bool = False,
) -> gym.Env:
    return make_masked_video_env(
        env_id, video_folder, seed=seed, mask_risky=mask_risky
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_masked_model(
    model: MaskablePPO,
    env_id: str,
    seed: int = 0,
    n_eval_episodes: int = 20,
) -> dict:
    """
    Evaluate a trained MaskablePPO model.

    The evaluation env also uses masking so the safety constraint is enforced
    at test time, matching real deployment behaviour.
    """
    rewards    = []
    lengths    = []
    violations = []
    successes  = []
    masked_unsafe_list = []

    env = make_masked_env(env_id, seed=seed)

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done   = False

        while not done:
            # Pass the action mask so MaskablePPO respects safety constraints
            # during evaluation (deterministic greedy policy).
            action_masks = env.action_masks()
            action, _ = model.predict(
                obs, deterministic=True, action_masks=action_masks
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                rewards.append(info.get("episode_reward", 0.0))
                lengths.append(info.get("episode_length", 0))
                violations.append(info.get("episode_violations", 0))
                successes.append(info.get("episode_success", 0))
                masked_unsafe_list.append(
                    info.get("masked_unsafe_attempts", 0)
                )

    env.close()

    return {
        "eval_mean_reward":          float(np.mean(rewards)),
        "eval_std_reward":           float(np.std(rewards)),
        "eval_mean_length":          float(np.mean(lengths)),
        "eval_mean_violations":      float(np.mean(violations)),
        "eval_success_rate":         float(np.mean(successes)),
        "eval_mean_masked_unsafe":   float(np.mean(masked_unsafe_list)),
    }


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

def summarize_training(
    csv_path: str,
    convergence_episode,
    convergence_timestep,
) -> dict:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {
            "episodes_logged":           0,
            "mean_reward":               float("nan"),
            "final_reward_mean_10":      float("nan"),
            "mean_length":               float("nan"),
            "mean_violations":           float("nan"),
            "success_rate":              float("nan"),
            "mean_masked_unsafe":        float("nan"),
            "convergence_episode":       None,
            "convergence_timestep":      None,
        }
    return {
        "episodes_logged":          int(len(df)),
        "mean_reward":              float(df["episode_reward"].mean()),
        "final_reward_mean_10":     float(df["episode_reward"].tail(10).mean()),
        "mean_length":              float(df["episode_length"].mean()),
        "mean_violations":          float(df["episode_violations"].mean()),
        "success_rate":             float(df["episode_success"].mean()),
        "mean_masked_unsafe":       float(df["masked_unsafe_attempts"].mean()),
        "convergence_episode":      convergence_episode,
        "convergence_timestep":     convergence_timestep,
    }


def record_video(
    env_id: str,
    model_path: str,
    out_dir: str,
    seed: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    env   = make_video_env(env_id, out_dir, seed=seed)
    model = MaskablePPO.load(model_path, device="cpu")
    obs, _ = env.reset()
    for _ in range(MAX_EVAL_STEPS):
        action_masks = env.action_masks()
        action, _ = model.predict(
            obs, deterministic=True, action_masks=action_masks
        )
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for d in (BASE_DIR, MODELS_DIR, VIDEOS_DIR):
        os.makedirs(d, exist_ok=True)

    summary_rows: list[dict] = []

    seed_csv_paths: dict[str, list[str]] = {
        env_id.replace("MiniGrid-", "").replace("-v0", "").lower(): []
        for env_id in ENV_IDS
    }

    for seed in SEEDS:
        set_seed(seed)

        for env_id in ENV_IDS:
            env_tag = env_id.replace("MiniGrid-", "").replace("-v0", "").lower()

            env_out_dir = os.path.join(BASE_DIR, env_tag, f"seed_{seed}")
            os.makedirs(env_out_dir, exist_ok=True)

            csv_path   = os.path.join(env_out_dir, "training_log.csv")
            model_path = os.path.join(MODELS_DIR, f"{env_tag}_seed{seed}_masked_ppo.zip")
            seed_csv_paths[env_tag].append(csv_path)

            print(f"\n=== Training (Masked PPO) | {env_id} | seed={seed} ===")

            env       = make_train_env(env_id, seed)
            logger_cb = MaskedEpisodeCSVLogger(csv_path)

            model = MaskablePPO(env=env, seed=seed, **PPO_KWARGS)
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger_cb)
            model.save(model_path)

            # LavaMaskingWrapper is two wrappers deep (ActionMasker wraps it)
            # Reach it via env.env to retrieve visit_counts.
            masking_wrapper = env.env  # ActionMasker.env → LavaMaskingWrapper
            visit_counts    = masking_wrapper.visit_counts
            np.save(os.path.join(env_out_dir, "visit_counts.npy"), visit_counts)
            plot_visit_heatmap(
                visit_counts,
                os.path.join(env_out_dir, "visit_heatmap.png"),
                title=f"{env_id} seed={seed} - Masked PPO Visitation",
            )

            env.close()

            plot_training_curves(
                csv_path=csv_path,
                out_dir=env_out_dir,
                title_prefix=f"{env_id} seed={seed} (Masked PPO)",
            )

            model_loaded = MaskablePPO.load(model_path, device="cpu")
            eval_stats   = evaluate_masked_model(
                model_loaded, env_id, seed=seed, n_eval_episodes=N_EVAL_EPISODES
            )

            record_video(
                env_id,
                model_path,
                out_dir=os.path.join(VIDEOS_DIR, env_tag, f"seed_{seed}"),
                seed=seed,
            )

            train_stats = summarize_training(
                csv_path,
                convergence_episode  = logger_cb.convergence_episode,
                convergence_timestep = logger_cb.convergence_timestep,
            )

            row = {"env_id": env_id, "seed": seed}
            row.update(train_stats)
            row.update(eval_stats)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    numeric_cols = summary_df.select_dtypes(include="number").columns.difference(["seed"])
    agg_df = (
        summary_df
        .groupby("env_id")[numeric_cols]
        .agg(["mean", "std"])
    )
    agg_df.to_csv(AGG_CSV)

    for env_id in ENV_IDS:
        env_tag = env_id.replace("MiniGrid-", "").replace("-v0", "").lower()
        plot_aggregated_curves(
            seed_csv_paths=seed_csv_paths[env_tag],
            out_dir=os.path.join(BASE_DIR, env_tag, "aggregated"),
            title_prefix=f"{env_id} (Masked PPO)",
        )

    print("\nDone.")
    print(f"Per-seed summary  : {SUMMARY_CSV}")
    print(f"Aggregated summary: {AGG_CSV}")
    print(agg_df)


if __name__ == "__main__":
    main()
