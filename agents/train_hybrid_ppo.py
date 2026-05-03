"""
agents/train_hybrid_ppo.py
==========================
Hybrid variant 1: hard unsafe-action masking + risky-action penalty.

Compared methods:
  * Vanilla PPO: no safety intervention.
  * Penalty PPO: penalizes lava after the agent reaches it.
  * Masked PPO: blocks unsafe actions.
  * Hybrid PPO: blocks unsafe actions and penalizes risky-but-allowed actions.

Hybrid behavior:
  * Forward into lava is unavailable through MaskablePPO action masks.
  * Risky actions remain available and receive RISKY_PENALTY reward shaping.
"""

from __future__ import annotations

import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO

from env.lava_hybrid_wrapper import make_hybrid_env, make_hybrid_video_env
from metrics.hybrid_training_logger import HybridEpisodeCSVLogger
from metrics.plot_results import (
    plot_aggregated_curves,
    plot_training_curves,
    plot_visit_heatmap,
)


BASE_DIR = "results/hybrid_ppo"
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
SUMMARY_CSV = os.path.join(BASE_DIR, "summary.csv")
AGG_CSV = os.path.join(BASE_DIR, "summary_aggregated.csv")

RISKY_PENALTY = 0.1
RECORD_VIDEOS = False

ENV_IDS = [
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",
]

SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 300_000
MAX_EVAL_STEPS = 300
N_EVAL_EPISODES = 20

PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01,
    verbose=1,
    device="cpu",
)


def make_train_env(env_id: str, seed: int) -> gym.Env:
    return make_hybrid_env(env_id, seed=seed, risky_penalty=RISKY_PENALTY)


def make_video_env(env_id: str, video_folder: str, seed: int) -> gym.Env:
    return make_hybrid_video_env(
        env_id,
        video_folder,
        seed=seed,
        risky_penalty=RISKY_PENALTY,
    )


def evaluate_hybrid_model(
    model: MaskablePPO,
    env_id: str,
    seed: int = 0,
    n_eval_episodes: int = 20,
) -> dict:
    rewards = []
    lengths = []
    violations = []
    successes = []
    masked_unsafe_list = []
    risky_actions_list = []
    risky_penalty_list = []

    env = make_hybrid_env(env_id, seed=seed, risky_penalty=RISKY_PENALTY)

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                rewards.append(info.get("episode_reward", 0.0))
                lengths.append(info.get("episode_length", 0))
                violations.append(info.get("episode_violations", 0))
                successes.append(info.get("episode_success", 0))
                masked_unsafe_list.append(info.get("masked_unsafe_attempts", 0))
                risky_actions_list.append(info.get("risky_actions", 0))
                risky_penalty_list.append(info.get("risky_penalty_total", 0.0))

    env.close()

    return {
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_std_reward": float(np.std(rewards)),
        "eval_mean_length": float(np.mean(lengths)),
        "eval_mean_violations": float(np.mean(violations)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_mean_masked_unsafe": float(np.mean(masked_unsafe_list)),
        "eval_mean_risky_actions": float(np.mean(risky_actions_list)),
        "eval_mean_risky_penalty": float(np.mean(risky_penalty_list)),
    }


def summarize_training(
    csv_path: str,
    convergence_episode,
    convergence_timestep,
) -> dict:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {
            "episodes_logged": 0,
            "mean_reward": float("nan"),
            "final_reward_mean_10": float("nan"),
            "mean_length": float("nan"),
            "mean_violations": float("nan"),
            "success_rate": float("nan"),
            "mean_masked_unsafe": float("nan"),
            "mean_risky_actions": float("nan"),
            "mean_risky_penalty": float("nan"),
            "convergence_episode": None,
            "convergence_timestep": None,
        }

    return {
        "episodes_logged": int(len(df)),
        "mean_reward": float(df["episode_reward"].mean()),
        "final_reward_mean_10": float(df["episode_reward"].tail(10).mean()),
        "mean_length": float(df["episode_length"].mean()),
        "mean_violations": float(df["episode_violations"].mean()),
        "success_rate": float(df["episode_success"].mean()),
        "mean_masked_unsafe": float(df["masked_unsafe_attempts"].mean()),
        "mean_risky_actions": float(df["risky_actions"].mean()),
        "mean_risky_penalty": float(df["risky_penalty_total"].mean()),
        "convergence_episode": convergence_episode,
        "convergence_timestep": convergence_timestep,
    }


def record_video(env_id: str, model_path: str, out_dir: str, seed: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    env = make_video_env(env_id, out_dir, seed=seed)
    model = MaskablePPO.load(model_path, device="cpu")
    obs, _ = env.reset()

    for _ in range(MAX_EVAL_STEPS):
        action_masks = env.action_masks()
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=action_masks,
        )
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

            csv_path = os.path.join(env_out_dir, "training_log.csv")
            model_path = os.path.join(
                MODELS_DIR,
                f"{env_tag}_seed{seed}_hybrid_ppo.zip",
            )
            seed_csv_paths[env_tag].append(csv_path)

            print(f"\n=== Training (Hybrid PPO) | {env_id} | seed={seed} ===")

            env = make_train_env(env_id, seed)
            logger_cb = HybridEpisodeCSVLogger(csv_path)

            model = MaskablePPO(env=env, seed=seed, **PPO_KWARGS)
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger_cb)
            model.save(model_path)

            hybrid_wrapper = env.env
            visit_counts = hybrid_wrapper.visit_counts
            np.save(os.path.join(env_out_dir, "visit_counts.npy"), visit_counts)
            plot_visit_heatmap(
                visit_counts,
                os.path.join(env_out_dir, "visit_heatmap.png"),
                title=f"{env_id} seed={seed} - Hybrid PPO Visitation",
            )

            env.close()

            plot_training_curves(
                csv_path=csv_path,
                out_dir=env_out_dir,
                title_prefix=f"{env_id} seed={seed} (Hybrid PPO)",
            )

            model_loaded = MaskablePPO.load(model_path, device="cpu")
            eval_stats = evaluate_hybrid_model(
                model_loaded,
                env_id,
                seed=seed,
                n_eval_episodes=N_EVAL_EPISODES,
            )

            if RECORD_VIDEOS:
                record_video(
                    env_id,
                    model_path,
                    out_dir=os.path.join(VIDEOS_DIR, env_tag, f"seed_{seed}"),
                    seed=seed,
                )

            train_stats = summarize_training(
                csv_path,
                convergence_episode=logger_cb.convergence_episode,
                convergence_timestep=logger_cb.convergence_timestep,
            )

            row = {"env_id": env_id, "seed": seed}
            row.update(train_stats)
            row.update(eval_stats)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    numeric_cols = summary_df.select_dtypes(include="number").columns.difference(["seed"])
    agg_df = summary_df.groupby("env_id")[numeric_cols].agg(["mean", "std"])
    agg_df.to_csv(AGG_CSV)

    for env_id in ENV_IDS:
        env_tag = env_id.replace("MiniGrid-", "").replace("-v0", "").lower()
        plot_aggregated_curves(
            seed_csv_paths=seed_csv_paths[env_tag],
            out_dir=os.path.join(BASE_DIR, env_tag, "aggregated"),
            title_prefix=f"{env_id} (Hybrid PPO)",
        )

    print("\nDone.")
    print(f"Per-seed summary  : {SUMMARY_CSV}")
    print(f"Aggregated summary: {AGG_CSV}")
    print(agg_df)


if __name__ == "__main__":
    main()
