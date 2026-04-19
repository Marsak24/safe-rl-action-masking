import numpy as np
from minigrid.wrappers import FlatObsWrapper
import gymnasium as gym

from env.lava_logging_wrapper import LavaLoggingWrapper


def evaluate_model(model, env_id, seed=0, n_eval_episodes=20):
    rewards = []
    lengths = []
    violations = []
    successes = []

    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = FlatObsWrapper(env)
    env = LavaLoggingWrapper(env)

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                rewards.append(info.get("episode_reward", 0.0))
                lengths.append(info.get("episode_length", 0))
                violations.append(info.get("episode_violations", 0))
                successes.append(info.get("episode_success", 0))

    env.close()

    return {
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_std_reward": float(np.std(rewards)),
        "eval_mean_length": float(np.mean(lengths)),
        "eval_mean_violations": float(np.mean(violations)),
        "eval_success_rate": float(np.mean(successes)),
    }
