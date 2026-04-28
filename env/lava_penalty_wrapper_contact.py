from env.lava_logging_wrapper import LavaLoggingWrapper

class LavaPenaltyWrapper(LavaLoggingWrapper):
    def __init__(self, env, lava_penalty: float = 0.5):
        super().__init__(env)
        self.lava_penalty = lava_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        adjusted_reward = float(reward)
        self.episode_length += 1
        self._log_visit()

        if self._on_lava():
            self.episode_violations += 1
            adjusted_reward -= self.lava_penalty

        if self._on_goal():
            self.episode_success = 1

        self.episode_reward += adjusted_reward

        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = self.episode_reward
            info["episode_length"] = self.episode_length
            info["episode_violations"] = self.episode_violations
            info["episode_success"] = self.episode_success

        return obs, adjusted_reward, terminated, truncated, info
