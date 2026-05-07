"""
agents/train_masked_ppo_crossing.py
=====================================
Hard masking (Method 3) on MiniGrid LavaCrossing environments.

Identical mechanism to train_masked_ppo.py — only ENV_IDS and BASE_DIR
differ so results are kept separate and directly comparable.

LavaCrossing is a harder test than LavaGap: the agent MUST navigate through
gaps in horizontal lava rivers to reach the goal. The optimal path always
passes adjacent to lava, so risky actions cannot be avoided. This tests
whether hard masking's convergence advantage holds on denser lava layouts.

Environments
------------
  MiniGrid-LavaCrossingS9N1-v0  — 9×9 grid, 1 lava river to cross
  MiniGrid-LavaCrossingS9N2-v0  — 9×9 grid, 2 lava rivers
  MiniGrid-LavaCrossingS9N3-v0  — 9×9 grid, 3 lava rivers (hardest)

Results saved to results/masked_ppo_crossing/.

Run
---
    python -m agents.train_masked_ppo_crossing
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from env.lava_masking_wrapper import make_masked_env, make_masked_video_env
from metrics.masked_training_logger import MaskedEpisodeCSVLogger
from metrics.plot_results import (
    plot_training_curves,
    plot_visit_heatmap,
    plot_aggregated_curves,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = "results/masked_ppo_crossing"
MODELS_DIR  = os.path.join(BASE_DIR, "models")
VIDEOS_DIR  = os.path.join(BASE_DIR, "videos")
SUMMARY_CSV = os.path.join(BASE_DIR, "summary.csv")
AGG_CSV     = os.path.join(BASE_DIR, "summary_aggregated.csv")

MASK_RISKY = False  # hard masking only: block forward-into-lava, nothing else

ENV_IDS = [
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
]

SEEDS              = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS    = 1_000_000   # LavaCrossing needs ~3-5x more steps than LavaGap
MAX_EVAL_STEPS     = 500   # LavaCrossing episodes can be longer than LavaGap
N_EVAL_EPISODES    = 20
CHECKPOINT_FREQ    = 50_000

# Identical to all other methods for fair comparison.
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
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_masked_model(
    model: MaskablePPO,
    env_id: str,
    seed: int = 0,
    n_eval_episodes: int = 20,
) -> dict:
    rewards, lengths, violations, successes, masked_unsafe = [], [], [], [], []

    env = make_masked_env(env_id, seed=seed, mask_risky=MASK_RISKY)

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done   = False
        while not done:
            action_masks = env.action_masks()
            action, _    = model.predict(
                obs, deterministic=True, action_masks=action_masks
            )
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                rewards.append(info.get("episode_reward",       0.0))
                lengths.append(info.get("episode_length",       0))
                violations.append(info.get("episode_violations",   0))
                successes.append(info.get("episode_success",    0))
                masked_unsafe.append(info.get("masked_unsafe_attempts", 0))

    env.close()
    return {
        "eval_mean_reward":        float(np.mean(rewards)),
        "eval_std_reward":         float(np.std(rewards)),
        "eval_mean_length":        float(np.mean(lengths)),
        "eval_mean_violations":    float(np.mean(violations)),
        "eval_success_rate":       float(np.mean(successes)),
        "eval_mean_masked_unsafe": float(np.mean(masked_unsafe)),
    }


def record_video(env_id: str, model_path: str, out_dir: str, seed: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    env   = make_masked_video_env(env_id, out_dir, seed=seed, mask_risky=MASK_RISKY)
    model = MaskablePPO.load(model_path, device="cpu")
    obs, _ = env.reset()
    for _ in range(MAX_EVAL_STEPS):
        action_masks = env.action_masks()
        action, _    = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


def summarize_training(
    csv_path: str,
    convergence_episode,
    convergence_timestep,
) -> dict:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {k: float("nan") for k in [
            "episodes_logged", "mean_reward", "final_reward_mean_10",
            "mean_length", "mean_violations", "success_rate",
            "mean_masked_unsafe", "convergence_episode", "convergence_timestep",
        ]}
    return {
        "episodes_logged":      int(len(df)),
        "mean_reward":          float(df["episode_reward"].mean()),
        "final_reward_mean_10": float(df["episode_reward"].tail(10).mean()),
        "mean_length":          float(df["episode_length"].mean()),
        "mean_violations":      float(df["episode_violations"].mean()),
        "success_rate":         float(df["episode_success"].mean()),
        "mean_masked_unsafe":   float(df["masked_unsafe_attempts"].mean()),
        "convergence_episode":  convergence_episode,
        "convergence_timestep": convergence_timestep,
    }


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
            model_path = os.path.join(
                MODELS_DIR, f"{env_tag}_seed{seed}_masked_ppo.zip"
            )
            seed_csv_paths[env_tag].append(csv_path)

            print(f"\n=== Training (Hard Masked PPO) | {env_id} | seed={seed} ===")

            env       = make_masked_env(env_id, seed=seed, mask_risky=MASK_RISKY)
            logger_cb = MaskedEpisodeCSVLogger(csv_path)

            # Resume from checkpoint if available
            ckpt_dir = os.path.join(env_out_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            existing_ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")],
                key=lambda f: int(f.split("_steps")[0].split("_")[-1])
                if "_steps" in f else 0,
            )

            if existing_ckpts:
                latest_ckpt = os.path.join(ckpt_dir, existing_ckpts[-1])
                steps_done  = int(
                    existing_ckpts[-1].split("_steps")[0].split("_")[-1]
                )
                remaining   = max(0, TOTAL_TIMESTEPS - steps_done)
                print(
                    f"  Resuming from checkpoint: {latest_ckpt} "
                    f"({steps_done} steps done, {remaining} remaining)"
                )
                model = MaskablePPO.load(
                    latest_ckpt, env=env, device=PPO_KWARGS["device"]
                )
            else:
                steps_done = 0
                remaining  = TOTAL_TIMESTEPS
                model      = MaskablePPO(env=env, seed=seed, **PPO_KWARGS)

            if remaining > 0:
                checkpoint_cb = CheckpointCallback(
                    save_freq   = CHECKPOINT_FREQ,
                    save_path   = ckpt_dir,
                    name_prefix = f"{env_tag}_seed{seed}_masked_ppo",
                    verbose     = 0,
                )
                model.learn(
                    total_timesteps     = remaining,
                    callback            = [logger_cb, checkpoint_cb],
                    reset_num_timesteps = (steps_done == 0),
                )

            model.save(model_path)

            # Visit heatmap — ActionMasker → LavaMaskingWrapper
            masking_wrapper = env.env
            visit_counts    = masking_wrapper.visit_counts
            np.save(os.path.join(env_out_dir, "visit_counts.npy"), visit_counts)
            plot_visit_heatmap(
                visit_counts,
                os.path.join(env_out_dir, "visit_heatmap.png"),
                title=f"{env_id} seed={seed} - Hard Masked PPO Visitation",
            )
            env.close()

            plot_training_curves(
                csv_path=csv_path,
                out_dir=env_out_dir,
                title_prefix=f"{env_id} seed={seed} (Hard Masked PPO)",
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

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    numeric_cols = summary_df.select_dtypes(include="number").columns.difference(
        ["seed"]
    )
    agg_df = summary_df.groupby("env_id")[numeric_cols].agg(["mean", "std"])
    agg_df.to_csv(AGG_CSV)

    for env_id in ENV_IDS:
        env_tag = env_id.replace("MiniGrid-", "").replace("-v0", "").lower()
        plot_aggregated_curves(
            seed_csv_paths=seed_csv_paths[env_tag],
            out_dir=os.path.join(BASE_DIR, env_tag, "aggregated"),
            title_prefix=f"{env_id} (Hard Masked PPO)",
        )

    print("\nDone.")
    print(f"Per-seed summary  : {SUMMARY_CSV}")
    print(f"Aggregated summary: {AGG_CSV}")
    print(agg_df)


if __name__ == "__main__":
    main()
