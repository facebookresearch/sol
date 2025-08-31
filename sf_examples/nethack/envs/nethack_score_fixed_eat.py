# Copyright (c) Facebook, Inc. and its affiliates.
import enum
import gymnasium as gym
from gymnasium.envs import registration

import numpy as np
import re

from nle import nethack
from nle.env import base

from nle.env.base import ASCII_SPACE, ASCII_ESC

TASK_ACTIONS = tuple(
    [nethack.MiscAction.MORE]
    + list(nethack.CompassDirection)
    + list(nethack.CompassDirectionLonger)
    + list(nethack.MiscDirection)
    + [nethack.Command.KICK, nethack.Command.EAT, nethack.Command.SEARCH]
)


class NetHackScoreFixedEat(base.NLE):
    """
    This is a copy of the Score env, except that the EAT action causes a random edible 
    inventory item to be eaten. With the standard Score task, many times eating an item 
    is not possible because the action for selecting that inventory item is missing. 
    """

    def __init__(
        self,
        *args,
        penalty_mode="constant",
        penalty_step: float = -0.01,
        penalty_time: float = -0.0,
        **kwargs,
    ):
        self.penalty_mode = penalty_mode
        self.penalty_step = penalty_step
        self.penalty_time = penalty_time

        self._frozen_steps = 0
        self.eat_msg = "What do you want to eat"
        self.re_inv_pattern = re.compile(r"\[([a-zA-Z]+)")
        

        actions = kwargs.pop("actions", TASK_ACTIONS)
        super().__init__(*args, actions=actions, **kwargs)

    def _get_time_penalty(self, last_observation, observation):
        blstats_old = last_observation[self._blstats_index]
        blstats_new = observation[self._blstats_index]

        old_time = blstats_old[nethack.NLE_BL_TIME]
        new_time = blstats_new[nethack.NLE_BL_TIME]

        if old_time == new_time:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0

        penalty = 0
        if self.penalty_mode == "constant":
            if self._frozen_steps > 0:
                penalty += self.penalty_step
        elif self.penalty_mode == "exp":
            penalty += 2**self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "square":
            penalty += self._frozen_steps**2 * self.penalty_step
        elif self.penalty_mode == "linear":
            penalty += self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "always":
            penalty += self.penalty_step
        else:  # default
            raise ValueError("Unknown penalty_mode '%s'" % self.penalty_mode)
        penalty += (new_time - old_time) * self.penalty_time
        return penalty

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Score delta, but with added a state loop penalty."""
        score_diff = super()._reward_fn(
            last_observation, action, observation, end_status
        )
        time_penalty = self._get_time_penalty(last_observation, observation)
        return score_diff + time_penalty


    def step(self, action: int):
        """Steps the environment.

        Args:
            action (int): action integer as defined by ``self.action_space``.

        Returns:
            (dict, float, bool, dict): a tuple containing
                - (*dict*): an observation of the state; this will contain the keys
                  specified by ``self.observation_space``.
                - (*float*): a reward; see ``self._reward_fn`` to see how it is
                  specified.
                - (*bool*): True if the state is terminal, False otherwise.
                - (*bool*): True if the episode is truncated, False otherwise.
                - (*dict*): a dictionary of extra information (such as
                  `end_status`, i.e. a status info -- death, task win, etc. --
                  for the terminal state).
        """
        # Careful: By default we re-use Numpy arrays, so copy before!
        last_observation = tuple(a.copy() for a in self.last_observation)

        # Handle EAT action
        last_msg = bytes(last_observation[self._message_index]).decode("latin-1")
        if self.eat_msg in last_msg:
            match = self.re_inv_pattern.search(last_msg)
            if match and self.actions[action] == ord("y"):
                # Action 'y' for 'yes' will lead to eating any random item in the inventory
                action = ord(match.group(1)[0])
            else:
                # Otherwise escape
                action = ASCII_SPACE
        else:
            action = self.actions[action]


        observation, done = self.nethack.step(action)

        # from here on nothing is changed wrt original
        truncated = self._check_abort(observation)

        is_game_over = observation[self._program_state_index][0] == 1
        if is_game_over or not self._allow_all_modes:
            observation, done = self._perform_known_steps(
                observation, done, exceptions=True
            )

        self._steps += 1

        self.last_observation = observation

        end_status = self._get_end_status(observation, done)

        reward = float(
            self._reward_fn(last_observation, action, observation, end_status)
        )

        if end_status and not done:
            # Try to end the game nicely.
            self._quit_game(observation, done)
            done = True

        return (
            self._get_observation(observation),
            reward,
            done,
            truncated,
            self._get_information(end_status),
        )

    

    

registration.register(
    id="NetHackScoreFixedEat-v0",
    entry_point=NetHackScoreFixedEat,
)
