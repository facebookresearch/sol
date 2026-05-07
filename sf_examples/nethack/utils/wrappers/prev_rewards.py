import gymnasium as gym
import numpy as np


class PrevRewardsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = 0

        obs_spaces = {"prev_rewards": gym.spaces.Box(-1000.0, 1000.0, shape=(1,), dtype=np.float32)}
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def reset(self, **kwargs):
        self.prev_reward = 0.0
        obs, info = self.env.reset(**kwargs)
        obs["prev_rewards"] = np.array([self.prev_reward])
        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        self.prev_reward = reward
        obs["prev_rewards"] = np.array([self.prev_reward])
        return obs, reward, term, trun, info
