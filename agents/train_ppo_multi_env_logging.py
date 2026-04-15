import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
import minigrid

from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO

from env.lava_logging_wrapper import LavaLoggingWrapper
from metrics.training_logger import EpisodeCSVLogger
from metrics.plot_results import plot_training_curves, plot_visit_heatmap


BASE_DIR = "results/vanilla_ppo"
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
SUMMARY_CSV = os.path.join(BASE_DIR, "summary.csv")

ENV_IDS = [
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",
]
TARGET_EPISODES = 200
LEARN_CHUNK_TIMESTEPS = 2048
EVAL_STEPS = 300

def make_train_env(env_id: str):
    env = gym.make(env_id)
    env = LavaLoggingWrapper(env)
    env = FlatObsWrapper(env)
    return env


def make_video_env(env_id: str, video_folder: str):
    env = gym.make(env_id, render_mode="rgb_array")
    env = LavaLoggingWrapper(env)
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    env = FlatObsWrapper(env)
    return env


def evaluate_and_record(env_id: str, model_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    env = make_video_env(env_id, out_dir)
    model = PPO.load(model_path, device="cpu")

    obs, _ = env.reset()
    total_reward = 0.0
    total_steps = 0
    violations = 0
    success = 0

    for _ in range(EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_steps += 1

        if info.get("episode_violations") is not None:
            violations = info["episode_violations"]
            success = info["episode_success"]

        if terminated or truncated:
            break

    env.close()

    eval_stats = {
        "eval_reward": total_reward,
        "eval_steps": total_steps,
        "eval_violations": violations,
        "eval_success": success,
    }

    with open(os.path.join(out_dir, "eval_stats.json"), "w") as f:
        json.dump(eval_stats, f, indent=2)

    return eval_stats


def summarize_training(csv_path: str):
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return {
            "episodes_logged": 0,
            "mean_reward": np.nan,
            "final_reward_mean_10": np.nan,
            "mean_length": np.nan,
            "mean_violations": np.nan,
            "success_rate": np.nan,
        }

    return {
        "episodes_logged": int(len(df)),
        "mean_reward": float(df["episode_reward"].mean()),
        "final_reward_mean_10": float(df["episode_reward"].tail(10).mean()),
        "mean_length": float(df["episode_length"].mean()),
        "mean_violations": float(df["episode_violations"].mean()),
        "success_rate": float(df["episode_success"].mean()),
    }


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    summary_rows = []

    for env_id in ENV_IDS:
        env_tag = env_id.replace("MiniGrid-", "").replace("-v0", "").lower()
        env_out_dir = os.path.join(BASE_DIR, env_tag)
        os.makedirs(env_out_dir, exist_ok=True)

        csv_path = os.path.join(env_out_dir, "training_log.csv")
        model_path = os.path.join(MODELS_DIR, f"{env_tag}_ppo.zip")

        print(f"\n=== Training on {env_id} ===")

        env = make_train_env(env_id)
        logger_cb = EpisodeCSVLogger(csv_path)


        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device="cpu",
        )

        while logger_cb.rows_written < TARGET_EPISODES:
            model.learn(
                total_timesteps=LEARN_CHUNK_TIMESTEPS,
                callback=logger_cb,
                reset_num_timesteps=False,
            )

        model.save(model_path)
        # Save visit heatmap
        visit_counts = env.env.global_visit_counts  # unwrap FlatObsWrapper -> LavaLoggingWrapper
        np.save(os.path.join(env_out_dir, "visit_counts.npy"), visit_counts)
        plot_visit_heatmap(
            visit_counts,
            os.path.join(env_out_dir, "visit_heatmap.png"),
            title=f"{env_id} - State Visitation Heatmap",
        )

        env.close()

        # Training plots
        plot_training_curves(
            csv_path=csv_path,
            out_dir=env_out_dir,
            title_prefix=env_id,
        )

        # Summary
        train_stats = summarize_training(csv_path)

        # Video + deterministic evaluation
        eval_stats = evaluate_and_record(
            env_id=env_id,
            model_path=model_path,
            out_dir=os.path.join(VIDEOS_DIR, env_tag),
        )

        row = {"env_id": env_id}
        row.update(train_stats)
        row.update(eval_stats)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\nDone. Summary saved to:", SUMMARY_CSV)
    print(summary_df)


if __name__ == "__main__":
    main()
