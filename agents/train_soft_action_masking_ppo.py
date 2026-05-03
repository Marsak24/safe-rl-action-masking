"""
agents/train_soft_action_masking_ppo.py
========================================
Method 6 — True Soft Action Masking:
  MaskablePPO + SoftMaskableActorCriticPolicy, where the environment's
  action_masks() returns FLOAT logit adjustments (not binary masks).

Mechanism
---------
  - No action is ever hard-blocked.
  - At every rollout step, SoftMaskableActorCriticPolicy adds the
    float logit adjustments from the wrapper to the raw policy logits
    before sampling.  Unsafe / risky actions are made probabilistically
    unlikely (but still selectable).
  - Violations CAN occur during training; they are recorded in the CSV.

This is distinct from:
  - Hard masking (Method 3): binary block, prob = 0
  - Hybrid masking (Method 4/5): binary block + reward penalty
  - Soft action masking (this file): float logit adjustment, no hard block

Results saved to results/soft_action_masked_ppo/.

Run
---
    python -m agents.train_soft_action_masking_ppo
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
BASE_DIR    = "results/soft_action_masked_ppo"
MODELS_DIR  = os.path.join(BASE_DIR, "models")
VIDEOS_DIR  = os.path.join(BASE_DIR, "videos")
SUMMARY_CSV = os.path.join(BASE_DIR, "summary.csv")
AGG_CSV     = os.path.join(BASE_DIR, "summary_aggregated.csv")

# Logit penalties for the soft action mask.
# Unsafe actions (forward into lava):  exp(5) ≈ 150× less likely than safe action
# Risky actions  (adjacent-to-lava):   exp(2) ≈   7× less likely than safe action
UNSAFE_LOGIT_PENALTY = 5.0
RISKY_LOGIT_PENALTY  = 2.0

ENV_IDS = [
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",
]

SEEDS              = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS    = 300_000
MAX_EVAL_STEPS     = 300
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
    """
    Guard: ensure the loaded model uses SoftMaskableCategoricalDistribution.
    If the saved policy was reconstructed as MaskableActorCriticPolicy (e.g.
    due to a custom-class lookup failure on load), swap in the correct
    distribution so that float logit adjustments are applied properly.
    """
    if not isinstance(model.policy.action_dist, SoftMaskableCategoricalDistribution):
        model.policy.action_dist = SoftMaskableCategoricalDistribution(
            int(model.policy.action_space.n)
        )
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: MaskablePPO,
    env_id: str,
    seed: int = 0,
    n_eval_episodes: int = 20,
) -> dict:
    """Run the model for n_eval_episodes and return aggregated stats."""
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
                MODELS_DIR, f"{env_tag}_seed{seed}_soft_action_masked_ppo.zip"
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
                    MaskablePPO.load(latest_ckpt, env=env, device=PPO_KWARGS["device"])
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
