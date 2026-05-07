from env.lava_logging_wrapper import LavaLoggingWrapper

class LavaPenaltyWrapper(LavaLoggingWrapper):
    def __init__(
        self,
        env,
        lava_penalty: float = 0.3,
        use_adjacent_penalty: bool = False,
        adjacent_penalty: float = 0.01,
    ):
        super().__init__(env)
        self.lava_penalty = lava_penalty
        self.use_adjacent_penalty = use_adjacent_penalty
        self.adjacent_penalty = adjacent_penalty

    def _adjacent_to_lava(self):
        base_env = self.unwrapped
        x, y = base_env.agent_pos
        neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        for nx, ny in neighbors:
            cell = base_env.grid.get(nx, ny)
            if cell is not None and getattr(cell, "type", None) == "lava":
                return True
        return False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        adjusted_reward = float(reward)
        if self._on_lava():
            adjusted_reward -= self.lava_penalty

        elif self.use_adjacent_penalty and self._adjacent_to_lava():
            adjusted_reward -= self.adjacent_penalty

        self.episode_reward += adjusted_reward
        self.episode_length += 1
        self._log_visit()

        if self._on_lava():
            self.episode_violations += 1

        if self._on_goal():
            self.episode_success = 1

        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = self.episode_reward
            info["episode_length"] = self.episode_length
            info["episode_violations"] = self.episode_violations
            info["episode_success"] = self.episode_success

        return obs, adjusted_reward, terminated, truncated, info