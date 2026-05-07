# Copyright (c) Facebook, Inc. and its affiliates.
import os

import gymnasium
import gymnasium as gym
from gymnasium.envs import registration

from minihack import MiniHack
from minihack.envs import register
from nle import nethack

from sf_examples.nethack.envs.nethack_score_fixed_eat import NetHackScoreFixedEat
from sf_examples.nethack.utils.task_rewards import (
    ScoreScore,
    HealthScore,
    StaircaseScore,
    GoldScore,
    ArmorScore, 
)


DAT_DIR = os.path.join(os.path.dirname(__file__), 'dat')


MOVE_ACTIONS = tuple(nethack.CompassDirection)
        
        
class MiniHackZombieHorde(NetHackScoreFixedEat, MiniHack):
    def __init__(self, *args, **kwargs):
        
        actions = MOVE_ACTIONS + (
            nethack.Command.SEARCH,
            nethack.Command.EAT,
        )
        
        super().__init__(
            *args,
            des_file=f"{DAT_DIR}/zombie_horde.des",
            autopickup=False,
            allow_all_modes=False,
            actions=actions,
            **kwargs
        )
        self.score = ScoreScore()

    def _reward_fn(self, last_observation, action, observation, end_status):
        return self.score.reward(self, last_observation, observation, end_status)
        
        

class MiniHackTreasureDash(MiniHack):
    def __init__(self, *args, **kwargs):
        
        actions = MOVE_ACTIONS + (
            nethack.Command.SEARCH,
        )        
        super().__init__(
            *args,
            des_file=f"{DAT_DIR}/treasure_dash.des",
            autopickup=True,
            allow_all_modes=False,
            actions=actions,
            **kwargs
        )
        self.gold = GoldScore()
        self.staircase = StaircaseScore()
        
    def _reward_fn(self, last_observation, action, observation, end_status):
        gold = self.gold.reward(self, last_observation, observation, end_status)
        staircase = self.staircase.reward(self, last_observation, observation, end_status)
        return 20 * staircase + gold



        
registration.register(
    id="MiniHack-ZombieHorde-v0",
    entry_point=MiniHackZombieHorde,
    kwargs={'character': "mon-hum-neu-mal", 'max_episode_steps': 1500}    
)

registration.register(
    id="MiniHack-TreasureDash-v0",
    entry_point=MiniHackTreasureDash,
    kwargs={'character': "mon-hum-neu-mal", 'max_episode_steps': 40}    
)




if __name__ == "__main__":
    import gymnasium as gym
    import minihack

    env1=gym.make('MiniHack-ZombieHorde-v0')
    env2=gym.make('MiniHack-TreasureDash-v0')


