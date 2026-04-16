import numpy as np
from minigrid.wrappers import FlatObsWrapper
import gymnasium as gym

from env.lava_logging_wrapper import LavaStatsWrapper


def make_eval_env(env_id, seed):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = LavaStatsWrapper(env)
    env = FlatObsWrapper(env)
    return env


def evaluate_model(model, env_id, seed, n_eval_episodes=20):
    rewards = []
    lengths = []
    violations = []
    successes = []

    for ep in range(n_eval_episodes):
        env = make_eval_env(env_id, seed + 1000 + ep)
        obs, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        rewards.append(env.unwrapped.ep_reward)
        lengths.append(env.unwrapped.ep_length)
        violations.append(env.unwrapped.ep_violations)
        successes.append(env.unwrapped.ep_success)
        env.close()

    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_violations_mean": float(np.mean(violations)),
        "eval_success_rate": float(np.mean(successes)),
    }
