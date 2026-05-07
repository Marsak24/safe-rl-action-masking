"""
agents/train_soft_action_masking_ppo_crossing.py
==================================================
Soft action masking (Method 6) on MiniGrid LavaCrossing environments.

Identical mechanism to train_soft_action_masking_ppo.py — only ENV_IDS and
BASE_DIR differ so results are kept separate and directly comparable to
train_masked_ppo_crossing.py.

LavaCrossing stress-tests soft action masking because:
  - The optimal path passes through the gap in each lava river.
  - Those gap cells are adjacent to lava → classified as risky.
  - A logit penalty on risky actions probabilistically discourages the
    agent from taking the very actions it NEEDS to reach the goal.
  - If RISKY_LOGIT_PENALTY is too large the agent will fail to converge,
    analogous to how hybrid penalty=0.1 collapsed on LavaGapS7.

This makes LavaCrossing a critical probe for tuning the risky penalty.

Environments
------------
  MiniGrid-LavaCrossingS9N1-v0  — 9×9 grid, 1 lava river to cross
  MiniGrid-LavaCrossingS9N2-v0  — 9×9 grid, 2 lava rivers
  MiniGrid-LavaCrossingS9N3-v0  — 9×9 grid, 3 lava rivers (hardest)

Results saved to results/soft_action_masked_ppo_crossing/.

Run
---
    python -m agents.train_soft_action_masking_ppo_crossing
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

from agents.soft_maskable_ppo import (
    SoftMaskableActorCriticPolicy,
    SoftMaskableCategoricalDistribution,
)
from env.lava_soft_action_masking_wrapper import (
    make_soft_action_masked_env,
    make_soft_action_masked_video_env,
)
from metrics.masked_training_logger import MaskedEpisodeCSVLogger
from metrics.plot_results import (
    plot_training_curves,
    plot_visit_heatmap,
    plot_aggregated_curves,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = "results/soft_action_masked_ppo_crossing"
MODELS_DIR  = os.path.join(BASE_DIR, "models")
VIDEOS_DIR  = os.path.join(BASE_DIR, "videos")
SUMMARY_CSV = os.path.join(BASE_DIR, "summary.csv")
AGG_CSV     = os.path.join(BASE_DIR, "summary_aggregated.csv")

# Logit penalties — tune these if the agent fails to converge.
# On LavaCrossing the risky penalty may need to be lower than on LavaGap
# because the optimal path necessarily traverses risky (lava-adjacent) cells.
# Start with the same values as train_soft_action_masking_ppo.py; reduce
# RISKY_LOGIT_PENALTY first if N2/N3 collapse.
UNSAFE_LOGIT_PENALTY = 5.0   # forward into lava:  exp(5) ≈ 150× less likely
RISKY_LOGIT_PENALTY  = 2.0   # adjacent to lava:   exp(2) ≈   7× less likely

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


def _ensure_soft_distribution(model: MaskablePPO) -> MaskablePPO:
    """Swap in SoftMaskableCategoricalDistribution if checkpoint loading lost it."""
    if not isinstance(model.policy.action_dist, SoftMaskableCategoricalDistribution):
        model.policy.action_dist = SoftMaskableCategoricalDistribution(
            int(model.policy.action_space.n)
        )
    return model


def evaluate_model(
    model: MaskablePPO,
    env_id: str,
    seed: int = 0,
    n_eval_episodes: int = 20,
) -> dict:
    rewards, lengths, violations, successes = [], [], [], []

    env = make_soft_action_masked_env(
        env_id,
        seed=seed,
        unsafe_logit_penalty=UNSAFE_LOGIT_PENALTY,
        risky_logit_penalty=RISKY_LOGIT_PENALTY,
    )

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
                rewards.append(info.get("episode_reward",     0.0))
                lengths.append(info.get("episode_length",     0))
                violations.append(info.get("episode_violations", 0))
                successes.append(info.get("episode_success",  0))

    env.close()
    return {
        "eval_mean_reward":     float(np.mean(rewards)),
        "eval_std_reward":      float(np.std(rewards)),
        "eval_mean_length":     float(np.mean(lengths)),
        "eval_mean_violations": float(np.mean(violations)),
        "eval_success_rate":    float(np.mean(successes)),
    }


def record_video(env_id: str, model_path: str, out_dir: str, seed: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    env   = make_soft_action_masked_video_env(
        env_id,
        out_dir,
        seed=seed,
        unsafe_logit_penalty=UNSAFE_LOGIT_PENALTY,
        risky_logit_penalty=RISKY_LOGIT_PENALTY,
    )
    model = _ensure_soft_distribution(MaskablePPO.load(model_path, device="cpu"))
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
            "convergence_episode", "convergence_timestep",
        ]}
    return {
        "episodes_logged":      int(len(df)),
        "mean_reward":          float(df["episode_reward"].mean()),
        "final_reward_mean_10": float(df["episode_reward"].tail(10).mean()),
        "mean_length":          float(df["episode_length"].mean()),
        "mean_violations":      float(df["episode_violations"].mean()),
        "success_rate":         float(df["episode_success"].mean()),
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
                MODELS_DIR,
                f"{env_tag}_seed{seed}_soft_action_masked_ppo.zip",
            )
            seed_csv_paths[env_tag].append(csv_path)

            print(
                f"\n=== Training (Soft Action Masked PPO) | {env_id} | seed={seed} ==="
            )

            env       = make_soft_action_masked_env(
                env_id,
                seed=seed,
                unsafe_logit_penalty=UNSAFE_LOGIT_PENALTY,
                risky_logit_penalty=RISKY_LOGIT_PENALTY,
            )
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
                remaining = max(0, TOTAL_TIMESTEPS - steps_done)
                print(
                    f"  Resuming from checkpoint: {latest_ckpt} "
                    f"({steps_done} steps done, {remaining} remaining)"
                )
                model = _ensure_soft_distribution(
                    MaskablePPO.load(
                        latest_ckpt, env=env, device=PPO_KWARGS["device"]
                    )
                )
            else:
                steps_done = 0
                remaining  = TOTAL_TIMESTEPS
                model = MaskablePPO(
                    policy = SoftMaskableActorCriticPolicy,
                    env    = env,
                    seed   = seed,
                    **PPO_KWARGS,
                )

            if remaining > 0:
                checkpoint_cb = CheckpointCallback(
                    save_freq   = CHECKPOINT_FREQ,
                    save_path   = ckpt_dir,
                    name_prefix = f"{env_tag}_seed{seed}_soft_action_masked_ppo",
                    verbose     = 0,
                )
                model.learn(
                    total_timesteps     = remaining,
                    callback            = [logger_cb, checkpoint_cb],
                    reset_num_timesteps = (steps_done == 0),
                )

            model.save(model_path)

            # Visit heatmap — ActionMasker → LavaSoftActionMaskingWrapper
            masking_wrapper = env.env
            visit_counts    = masking_wrapper.visit_counts
            np.save(os.path.join(env_out_dir, "visit_counts.npy"), visit_counts)
            plot_visit_heatmap(
                visit_counts,
                os.path.join(env_out_dir, "visit_heatmap.png"),
                title=f"{env_id} seed={seed} - Soft Action Masked PPO Visitation",
            )
            env.close()

            plot_training_curves(
                csv_path=csv_path,
                out_dir=env_out_dir,
                title_prefix=f"{env_id} seed={seed} (Soft Action Masked PPO)",
            )

            model_loaded = _ensure_soft_distribution(
                MaskablePPO.load(model_path, device="cpu")
            )
            eval_stats = evaluate_model(
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
            title_prefix=f"{env_id} (Soft Action Masked PPO)",
        )

    print("\nDone.")
    print(f"Per-seed summary  : {SUMMARY_CSV}")
    print(f"Aggregated summary: {AGG_CSV}")
    print(agg_df)


if __name__ == "__main__":
    main()
