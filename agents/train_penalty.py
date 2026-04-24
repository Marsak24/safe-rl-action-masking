import os
import json
import random

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from metrics.evaluation import evaluate_model
from env.lava_penalty_wrapper import LavaPenaltyWrapper
from metrics.training_logger import EpisodeCSVLogger
from metrics.plot_results import (
    plot_training_curves,
    plot_visit_heatmap,
    plot_aggregated_curves,
)


BASE_DIR    = "results/penalty_ppo"
MODELS_DIR  = os.path.join(BASE_DIR, "models")
VIDEOS_DIR  = os.path.join(BASE_DIR, "videos")
SUMMARY_CSV = os.path.join(BASE_DIR, "summary.csv")
AGG_CSV     = os.path.join(BASE_DIR, "summary_aggregated.csv")
LAVA_PENALTY = 0.5

ENV_IDS = [
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",
]

SEEDS          =  [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 300_000
MAX_EVAL_STEPS  = 300 #300
N_EVAL_EPISODES = 20 #20

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



def make_train_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = FlatObsWrapper(env)
    env = LavaPenaltyWrapper(env, lava_penalty=LAVA_PENALTY)

    return env


def make_video_env(env_id: str, video_folder: str, seed: int):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env = FlatObsWrapper(env)
    env = LavaPenaltyWrapper(env, lava_penalty=LAVA_PENALTY)
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    return env



def record_video(env_id: str, model_path: str, out_dir: str, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    env = make_video_env(env_id, out_dir, seed=seed)
    model = PPO.load(model_path, device="cpu")
    obs, _ = env.reset()
    for _ in range(MAX_EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


def summarize_training(csv_path: str, convergence_episode, convergence_timestep):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {
            "episodes_logged": 0,
            "mean_reward": float("nan"),
            "final_reward_mean_10": float("nan"),
            "mean_length": float("nan"),
            "mean_violations": float("nan"),
            "success_rate": float("nan"),
            "convergence_episode": None,
            "convergence_timestep": None,
        }
    return {
        "episodes_logged":        int(len(df)),
        "mean_reward":            float(df["episode_reward"].mean()),
        "final_reward_mean_10":   float(df["episode_reward"].tail(10).mean()),
        "mean_length":            float(df["episode_length"].mean()),
        "mean_violations":        float(df["episode_violations"].mean()),
        "success_rate":           float(df["episode_success"].mean()),
        "convergence_episode":    convergence_episode,   # from EpisodeCSVLogger
        "convergence_timestep":   convergence_timestep,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    for d in (BASE_DIR, MODELS_DIR, VIDEOS_DIR):
        os.makedirs(d, exist_ok=True)

    summary_rows = []

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
            model_path = os.path.join(MODELS_DIR, f"{env_tag}_seed{seed}_ppo.zip")
            seed_csv_paths[env_tag].append(csv_path)

            print(f"\n=== Training | {env_id} | seed={seed} ===")

            env        = make_train_env(env_id, seed)
            logger_cb  = EpisodeCSVLogger(csv_path)

            model = PPO(env=env, seed=seed, **PPO_KWARGS)
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger_cb)
            model.save(model_path)

            visit_counts = env.visit_counts        
            np.save(os.path.join(env_out_dir, "visit_counts.npy"), visit_counts)
            plot_visit_heatmap(
                visit_counts,
                os.path.join(env_out_dir, "visit_heatmap.png"),
                title=f"{env_id} seed={seed} - Visitation Heatmap",
            )

            env.close()

            plot_training_curves(
                csv_path=csv_path,
                out_dir=env_out_dir,
                title_prefix=f"{env_id} seed={seed}",
            )

            model_loaded = PPO.load(model_path, device="cpu")
            eval_stats   = evaluate_model(model_loaded, env_id, seed=seed,
                                          n_eval_episodes=N_EVAL_EPISODES)

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
            title_prefix=env_id,
        )

    print("\nDone.")
    print(f"Per-seed summary  : {SUMMARY_CSV}")
    print(f"Aggregated summary: {AGG_CSV}")
    print(agg_df)


if __name__ == "__main__":
    main()
