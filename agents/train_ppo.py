import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper

from stable_baselines3 import PPO

env = gym.make("MiniGrid-LavaGapS5-v0", render_mode="human")
env = FlatObsWrapper(env)

model = PPO("MlpPolicy", env, verbose=1, device="cpu")

model.learn(total_timesteps=10000)
model.save("ppo_lavagap")

print("Training done")
