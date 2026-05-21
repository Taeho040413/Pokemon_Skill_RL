import ast

import numpy as np
import gymnasium

import pufferlib.utils


class EpisodeStatsWrapper(gymnasium.Wrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env)
        self.reset()

    # Keep Gymnasium-compatible reset signature for vectorized workers.
    def reset(self, *, seed=None, options=None):
        self.info = dict(episode_return=0, episode_length=0)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        if not info:
            self.info["episode_return"] += reward
            self.info["episode_length"] += 1
            out = {}
            if (
                terminated
                or truncated
                or self.info["episode_length"] % self.env.log_frequency == 0
            ):
                out = self.info
            return observation, reward, terminated, truncated, out

        for k, v in pufferlib.utils.unroll_nested_dict(info):
            if "exploration_map" in k:
                self.info[k] = self.info.get(k, np.zeros_like(v)) + v
            elif k == "state" and isinstance(v, dict):
                # PyBoy save-state checkpoints only (not stats/rm_state, etc.).
                if "state" not in self.info:
                    self.info["state"] = {}
                self.info["state"] |= v
            elif isinstance(k, str) and k.startswith("state/"):
                if "state" not in self.info:
                    self.info["state"] = {}
                sub_key = k[6:]
                try:
                    state_key = ast.literal_eval(sub_key)
                except (SyntaxError, ValueError):
                    state_key = sub_key
                self.info["state"][state_key] = v
            else:
                self.info[k] = v

        self.info["episode_return"] += reward
        self.info["episode_length"] += 1

        out = {}
        if terminated or truncated or self.info["episode_length"] % self.env.log_frequency == 0:
            out = self.info

        return observation, reward, terminated, truncated, out
