import gymnasium as gym
import numpy as np
from collections import defaultdict



class MessageCountsWrapper(gym.Wrapper):
    """Keep some statistic about the messages."""

    def __init__(self, env):
        super().__init__(env)
        self.messages_dict = defaultdict(int)

        obs_spaces = {
            "msg_count": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }

        obs_spaces.update(
            [(k, self.env.observation_space[k]) for k in self.env.observation_space]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        obs["msg_count"] = np.array([1.0]).astype(np.float32)
        msg = bytes(obs["message"])
        self.messages_dict[msg] += 1
        obs["msg_count"] = np.array([self.messages_dict[msg]]).astype(np.float32)
        return obs, reward, term, trun, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.messages_dict = defaultdict(int)

        obs["msg_count"] = np.array([1.0]).astype(np.float32)
        msg = bytes(obs["message"])
        self.messages_dict[msg] += 1
        obs["msg_count"] = np.array([self.messages_dict[msg]]).astype(np.float32)
        return obs, info
