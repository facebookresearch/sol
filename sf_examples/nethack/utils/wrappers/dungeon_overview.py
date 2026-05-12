import gymnasium as gym
import numpy as np

import nle_patched.nle as nle
from nle_patched.nle.nethack.actions import Command
from nle_patched.nle import nethack




class DungeonOverviewWrapper(gym.Wrapper):

    def __init__(self, env, max_rows=15, max_str_length=50):
        super().__init__(env)

        self.max_rows = max_rows
        self.max_str_length = max_str_length

        obs_spaces = {"overview_strs": gym.spaces.Box(0, 255, shape=(self.max_rows, self.max_str_length), dtype=np.uint8)}
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)

        self.landmark_patterns = {
            'gnomish_mines': 'gnomish mines',
            'minetown': 'many shops',
            'sokoban': 'sokoban',
            'big_room': 'big room',
            'rogue': 'primitive',
            'quest': 'quest',
            'fort_ludios': 'fort ludios',
            'castle': 'castle',
            'gehennom': 'gehennom',
        }


    def _update_overview_strs(self):
        overview_tty_chars = []
        state, done = self.env.unwrapped.nethack.step(Command.OVERVIEW)

        while state[self.env.unwrapped._internal_index][3] and not done: # xwaitforspace
            tty_chars = state[self.env.unwrapped._observation_keys.index("tty_chars")]
            overview_tty_chars.append(tty_chars[:-2, 24:24+self.max_str_length].copy())
            state, done = self.env.unwrapped.nethack.step(ord(' '))

        if len(overview_tty_chars) > 0:
            overview_tty_chars = np.concatenate(overview_tty_chars, axis=0)

            # this will be included in the observation, we limit the rows for speed reasons
            self.overview_strs = overview_tty_chars[:self.max_rows]

            # record visiting various landmarks for metrics
            overview_strs = bytes(overview_tty_chars).decode("latin-1").lower()
            for key, pattern in self.landmark_patterns.items():
                if pattern in overview_strs:
                    self.landmarks_visited[key] = 1

        return done


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_overview_strs()
        obs['overview_strs'] = self.overview_strs

        self.last_dlvl = obs['blstats'][nethack.NLE_BL_DLEVEL]

        self.landmarks_visited = {}
        for key in self.landmark_patterns.keys():
            self.landmarks_visited[key] = 0

        return obs, info

    def add_more_stats(self, info):
        extra_stats = info.get("episode_extra_stats", {})
        new_extra_stats = {f"visited_{k}": v for k, v in self.landmarks_visited.items()}
        return {**extra_stats, **new_extra_stats}


    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)

        dlvl = obs['blstats'][nethack.NLE_BL_DLEVEL]

        if not (term or trun):
            if dlvl != self.last_dlvl:
                term = self._update_overview_strs()

        self.last_dlvl = dlvl

        obs['overview_strs'] = self.overview_strs

        if term or trun:
            info["episode_extra_stats"] = self.add_more_stats(info)

        return obs, reward, term, trun, info
