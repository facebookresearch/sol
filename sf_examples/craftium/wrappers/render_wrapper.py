import gymnasium as gym
import numpy as np


class ImageRenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space["img"] = gym.spaces.Box(0, 255, (3, 84, 84), np.uint8)

                                    
    def render(self):
        return np.transpose(self.last_obs["img"], (1, 2, 0))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["img"] = np.transpose(obs["img"], (2, 0, 1))
        self.last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        obs["img"] = np.transpose(obs["img"], (2, 0, 1))
        self.last_obs = obs
            
        return obs, reward, term, trun, info
