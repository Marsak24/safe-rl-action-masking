import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO

env_ids = [
    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",
    "MiniGrid-LavaGapS7-v0",
]

for env_id in env_ids:
    print(f"\n=== Training on {env_id} ===")

    env = gym.make(env_id)
    env = FlatObsWrapper(env)

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=10000)

    save_name = env_id.replace("MiniGrid-", "").replace("-v0", "").lower()
    model.save(f"models/{save_name}_ppo")

    env.close()

print("Done.")
