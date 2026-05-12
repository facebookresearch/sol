from collections import namedtuple
import json
import os
import re

import gymnasium as gym

BLStats = namedtuple(
    "BLStats",
    "x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask align_bits",
)

with open(os.path.join(os.path.dirname(__file__), "achievements.json"), "r") as f:
    ACHIEVEMENTS = json.load(f)


class ExtraInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.patterns = {
            'buc': re.compile(r"(cursed|blessed|\(\d+:\d+\))")
        }

    def step(self, action):
        # because we will see done=True at the first timestep of the new episode
        # to properly calculate blstats at the end of the episode we need to keep the last_observation around
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, reward, term, trun, info = self.env.step(action)

        if term or trun:
            info["episode_extra_stats"] = self.add_more_stats(info, last_observation)

        return obs, reward, term, trun, info

    def add_more_stats(self, info, last_observation):
        extra_stats = info.get("episode_extra_stats", {})
        new_extra_stats = {
            "is_satiated": self._is_satiated(last_observation),
            "is_hungry": self._is_hungry(last_observation),
            "is_food_poisoned": self._is_food_poisoned(last_observation),
            "progress": self._compute_progress(last_observation),
        }
        return {**extra_stats, **new_extra_stats}

    def _is_satiated(self, observation):
        blstats = observation[self.env.unwrapped._blstats_index]
        satiated = blstats[21] == 0
        return satiated

    def _is_hungry(self, observation):
        blstats = observation[self.env.unwrapped._blstats_index]
        hungry = blstats[21] == 2
        weak = blstats[21] == 3
        fainting = blstats[21] == 4
        return hungry or weak or fainting

    def _is_food_poisoned(self, observation):
        blstats = observation[self.env.unwrapped._blstats_index]
        poisoned = blstats[25] == 8
        return poisoned

    def _compute_progress(self, observation):
        dlvl = observation[self.env.unwrapped._blstats_index][12]
        xp = observation[self.env.unwrapped._blstats_index][18]

        dlvl = f'Dlvl:{dlvl}'
        xp = f'Xp:{xp}'

        dlvl_progress = ACHIEVEMENTS.get(dlvl, 0)
        xp_progress = ACHIEVEMENTS.get(xp, 0)

        return max(dlvl_progress, xp_progress) * 100.0
