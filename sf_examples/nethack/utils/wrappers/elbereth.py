import gymnasium as gym
import numpy as np
import nle.nethack as nethack
from collections import defaultdict



class ElberethWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        message = bytes(obs["message"]).decode("latin-1")

        if term or trun:
            return obs, reward, term, trun, info

        # Macro-action for engraving Elbereth
        if self.env.actions[action] == nethack.Command.ENGRAVE and "What do you want to write with?" in message:

            
            # write with fingertip in dust

            if not (term or trun):
                obs, reward, term, trun, info = self.env.step(self.env.actions.index(ord("-")))
            message = bytes(obs["message"]).decode("latin-1")

            if term or trun:
                return obs, reward, term, trun, info
            
            if "Do you want to add to the current engraving?" in message:
                obs, reward, term, trun, info = self.env.step(self.env.actions.index(ord("n")))

            if term or trun:
                return obs, reward, term, trun, info
                
            obs, reward, term, trun, info = self.env.step(self.env.actions.index(ord("\r")))

            if term or trun:
                return obs, reward, term, trun, info
            
            for l in "Elbereth":
                obs, reward, term, trun, info = self.env.step(self.env.actions.index(ord(l)))

            if term or trun:
                return obs, reward, term, trun, info
                
            obs, reward, term, trun, info = self.env.step(self.env.actions.index(ord("\r")))

            if term or trun:
                return obs, reward, term, trun, info
            
            # check our work
            #obs, reward, term, trun, info = self.env.step(self.env.actions.index(ord(":")))
            #_, reward, term, trun, info = self.env.step(self.env.actions.index(ord("\r")))
            #print(bytes(obs["message"]).decode("latin-1"))
            
        return obs, reward, term, trun, info
