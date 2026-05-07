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


class MiniHackGemStack(MiniHack):
    def __init__(self, *args, **kwargs):
        actions = MOVE_ACTIONS + (
            nethack.Command.PICKUP,
            nethack.Command.APPLY, # a (b is already included as a move action)
            nethack.Command.ESC,
            nethack.MiscAction.MORE,
        )

        super().__init__(
            *args,
            des_file=f"{DAT_DIR}/gem_stack.des",
            autopickup=False,
            allow_all_modes=True,
            actions=actions,
            **kwargs
        )

        self.found_gem = False

    def _reward_fn(self, last_observation, action, observation, end_status):
        msg = bytes(observation[self._message_index]).decode('latin-1')

        if "a white gem named put me down." in msg and not self.found_gem:
            self.found_gem = True
            reward = -100
        elif "a white gem named pick me up." in msg and not self.found_gem:
            self.found_gem = True
            reward = 100
        else:
            reward = 0

        return reward

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        self.found_gem = False

        return obs, info

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
        return self.score.reward(self, last_observation, observation, {"end_status": end_status})
        
        

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
        gold = self.gold.reward(self, last_observation, observation, {"end_status": end_status})
        staircase = self.staircase.reward(self, last_observation, observation, {"end_status": end_status})
        return 20 * staircase + gold


class MiniHackRingInv(MiniHack):
    def __init__(self, *args, **kwargs):
        
        actions = MOVE_ACTIONS + (
            nethack.Command.SEARCH,
        )        
        super().__init__(
            *args,
            des_file=f"{DAT_DIR}/ring_inv.des",
            autopickup=False,
            allow_all_modes=False,
            actions=actions,
            **kwargs
        )
        self.gold = GoldScore()
        self.staircase = StaircaseScore()
        self.score = ScoreScore()

    def _print_msg(self, raw_obs):
        message = raw_obs[self._message_index]
        message = bytes(message[message != 0].tolist()).decode("latin-1")
        print(message)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        # pick up ring 
        raw_obs, _ = self.nethack.step(nethack.Command.PICKUP)
        message = raw_obs[self._message_index]
        message = bytes(message[message != 0].tolist()).decode("latin-1")
                
        # put it on 
        ring_letter = message[0]
        raw_obs, _ = self.nethack.step(nethack.Command.PUTON)
        
        # make sure to pick the right one
        raw_obs, _ = self.nethack.step(ord(ring_letter))
        
        # pick right finger
        raw_obs, _ = self.nethack.step(ord('r'))
        raw_obs, _ = self.nethack.step(nethack.CompassDirection.W)

        # pick up identify scroll
        raw_obs, _ = self.nethack.step(nethack.Command.PICKUP)
        message = raw_obs[self._message_index]
        message = bytes(message[message != 0].tolist()).decode("latin-1")

        # read it
        scroll_letter = message[0]
        raw_obs, _ = self.nethack.step(nethack.Command.READ)        
        raw_obs, _ = self.nethack.step(ord(scroll_letter))
        raw_obs, _ = self.nethack.step(ord(" "))
        raw_obs, _ = self.nethack.step(nethack.Command.SEARCH)

        obs = self._get_observation(raw_obs)

        '''
        # visualize inventory
        inv = obs['inv_strs']
        inv[:, -1] = ord('\n')
        inv = inv[inv != 0]
        print(bytes(inv).decode('latin-1'))
        '''

        self._last_observation = tuple(a.copy() for a in raw_obs)

        return obs, info

    
    def _reward_fn(self, last_observation, action, observation, end_status):
        score = self.score.reward(self, last_observation, observation, end_status)
        return score
    



class MiniHackArmorInv(MiniHack):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            des_file=f"{DAT_DIR}/armor_inv.des",
            autopickup=True,
            #allow_all_modes=False,
            **kwargs
        )

        self.armor = ArmorScore()

    def _reward_fn(self, last_observation, action, observation, end_status):
        armor = self.armor.reward(self, last_observation, observation, end_status)
        return armor
        
        

class MiniHackArmorUncursed(MiniHack):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            des_file=f"{DAT_DIR}/armor_uncursed.des",
            autopickup=False, 
            **kwargs
        )
        self.armor = ArmorScore()

    def _reward_fn(self, last_observation, action, observation, end_status):
        armor = self.armor.reward(self, last_observation, observation, end_status)
        return armor
        
        

class MiniHackArmorMixedCurse(MiniHack):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            des_file=f"{DAT_DIR}/armor_mixed_curse.des",
            autopickup=False, 
            **kwargs
        )
        self.armor = ArmorScore()

    def _reward_fn(self, last_observation, action, observation, end_status):
        armor = self.armor.reward(self, last_observation, observation, end_status)
        return armor
        
        
        
class MiniHackArmorEnchantScrolls(MiniHack):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            des_file=f"{DAT_DIR}/armor_and_scrolls.des",
            autopickup=False, 
            **kwargs
        )
        self.armor = ArmorScore()

    def _reward_fn(self, last_observation, action, observation, end_status):
        armor = self.armor.reward(self, last_observation, observation, end_status)
        return armor
        


    
registration.register(
    id="MiniHack-GemStack-v0",
    entry_point=MiniHackGemStack,
    kwargs={'character': "val-dwa-law-fem", 'max_episode_steps': 20}
)
        
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

registration.register(
    id="MiniHack-RingInv-v0",
    entry_point=MiniHackRingInv,
    kwargs={'character': "cav-hum-law-mal", 'max_episode_steps': 200}    
)

registration.register(
    id="MiniHack-ArmorInv-v0",
    entry_point=MiniHackArmorInv,
    kwargs={'character': "wiz-hum-neu-mal", 'max_episode_steps': 50}    
)


registration.register(
    id="MiniHack-ArmorUncursed-v0",
    entry_point=MiniHackArmorUncursed,
    kwargs={'character': "wiz-hum-neu-mal", 'max_episode_steps': 1000}    
)

registration.register(
    id="MiniHack-ArmorIdentify-v0",
    entry_point=MiniHackArmorMixedCurse,
    kwargs={'character': "wiz-hum-neu-mal", 'max_episode_steps': 1000}    
)

registration.register(
    id="MiniHack-ArmorEnchant-v0",
    entry_point=MiniHackArmorEnchantScrolls,
    kwargs={'character': "wiz-hum-neu-mal", 'max_episode_steps': 1000}    
)



if __name__ == "__main__":
    import gymnasium as gym
    import minihack

    env1=gym.make('MiniHack-ZombieHorde-v0')
    env2=gym.make('MiniHack-TreasureDash-v0')


