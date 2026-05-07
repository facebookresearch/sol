import gymnasium as gym
import numpy as np
import nle.nethack as nethack
from collections import defaultdict



class EnhanceSkillsWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        message = bytes(obs["message"]).decode("latin-1")

        if "You feel more confident in" in message:
            obs, reward, term, trun, info = self.env.step(self.env.unwrapped.actions.index(nethack.Command.ENHANCE))
            obs, reward, term, trun, info = self.env.step(self.env.unwrapped.actions.index(ord('a')))
            
        return obs, reward, term, trun, info
