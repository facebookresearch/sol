import gymnasium as gym
import numpy as np

from sample_factory.utils.fpdb import ForkedPdb


import nle_patched.nle as nle
from nle_patched.nle.nethack.actions import Command
from nle_patched.nle import nethack


ALIGNMENTS = {
    b"lawful": 0,
    b"neutral": 1,
    b"chaotic": 2,
}

GENDERS = {
    b"male": 0,
    b"female": 1,
}

RACES = {
    b"human": 0,
    b"elf": 1,
    b"elven": 1,
    b"dwarf": 2,
    b"dwarven": 2,
    b"gnome": 3,
    b"gnomish": 3,
    b"orc": 4,
    b"orcish": 4,
}

ROLES = {
    b"archeologist": 0,
    b"barbarian": 1,
    b"caveman": 2,
    b"cavewoman": 2,
    b"healer": 3,
    b"knight": 4,
    b"monk": 5,
    b"priest": 6,
    b"priestess": 6,
    b"ranger": 7,
    b"rogue": 8,
    b"samurai": 9,
    b"tourist": 10,
    b"valkyrie": 11,
    b"wizard": 12,
}

class ExtraObsWrapper(gym.Wrapper):

    def __init__(self, env, max_spells=3, max_spells_str_length=50):
        super().__init__(env)

        self.max_spells = max_spells
        self.max_spells_str_length = max_spells_str_length
        obs_spaces = {
            "spells_strs": gym.spaces.Box(0, 255, shape=(self.max_spells, self.max_spells_str_length), dtype=np.uint8),
            "role": gym.spaces.Box(0.0, 1.0, shape=(13,), dtype=np.float32),
            "race": gym.spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
            "align": gym.spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
        }
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.actions_list = list(self.env.unwrapped.actions)


    def _get_attributes(self):
        obs, reward, term, trun, info = self.env.step(self.actions_list.index(Command.ATTRIBUTES))
        gender, race, role = bytes(obs['tty_chars'][3]).strip().split(b' ')[-3:]
        alignment = bytes(obs['tty_chars'][4]).strip().split(b' ')[2]
        role = role.lower().replace(b'.', b'')
        alignment = alignment.replace(b',', b'')
        self.gender = np.zeros(2)
        self.role = np.zeros(13)
        self.race = np.zeros(5)
        self.alignment = np.zeros(3)

        self.role[ROLES[role.lower()]] = 1
        self.race[RACES[race.lower()]] = 1
        self.alignment[ALIGNMENTS[alignment]] = 1

        # skip gender for now, it makes logic more complex due to some roles having implicit gender
        # (Priest/Priestess) and others not (Samurai) and has almost no effect on the game.

        obs, reward, term, trun, info = self.env.step(self.actions_list.index(ord(' ')))
        obs, reward, term, trun, info = self.env.step(self.actions_list.index(ord(' ')))

        return obs, reward, term, trun, info


    def _get_spells_strs(self):
        obs, reward, term, trun, info = self.env.step(self.actions_list.index(Command.SEESPELLS))
        tty_chars = obs['tty_chars']
        self.spells_strs = tty_chars[
            3 : 3 + self.max_spells,
            24 : 24 + self.max_spells_str_length
        ].copy()
        obs, reward, term, trun, info = self.env.step(self.actions_list.index(ord(' ')))
        return obs, reward, term, trun, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, reward, term, trun, info = self._get_spells_strs()
        obs, reward, term, trun, info = self._get_attributes()

        obs['role'] = self.role
        obs['race'] = self.race
        obs['align'] = self.alignment
        obs['spells_strs'] = self.spells_strs

        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)

        obs['role'] = self.role
        obs['race'] = self.race
        obs['align'] = self.alignment
        obs['spells_strs'] = self.spells_strs

        return obs, reward, term, trun, info
