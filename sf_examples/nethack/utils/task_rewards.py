import re
import enum

import numpy as np
from sf_examples.nethack.utils.nle_tokenizer.tokenizer import MONSTERS, NLE_TOKENIZER, NLE_TOKENIZER_TUPLE_2_TOK, NLE_UNIQUE_TOKENS
from nle import nethack

projectile_pattern = re.compile(r'.*(dart|arrow|ya|bolt|dagger|rock|stone)\s+hits\s+.*')

PROJECTILES_SET = {'dart', 'arrow', 'ya', 'bolt', 'dagger', 'rock', 'stone', 'spell'}
MONSTERS_SET = set(MONSTERS)

INTRINSICS_ACQUIRED_MSGS = (
    "you feel healthy",
    "you feel especially healthy",
    "you feel wide awake",
    "you feel full of hot air",
    "you feel a momentary chill",
    "your health currently feels amplified",
    "you feel very firm",
    "you feel a strange mental acuity",
)

class Score:

    class StepStatus(enum.IntEnum):
        ABORTED = -1
        RUNNING = 0
        DEATH = 1
        TASK_SUCCESSFUL = 2
    
    def __init__(self):
        self.score = 0
        self.sokoban_levels = {}
        self.max_gold = 0
        # convert name to snake_case
        self.name = re.sub('(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', self.__class__.__name__).lower()


    def reset_score(self):
        self.score = 0


class OracleScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        message = bytes(observation[env._message_index]).decode("latin-1")

        reward = 0
        # ensure that we will give reward only once
        if self.score == 0:
            if "You're in Delphi" in message or "welcome to Delphi!" in message:
                self.score = 1
                reward = 1

        return reward




class IntrinsicsScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        message = bytes(observation[env._message_index]).decode("latin-1").lower()

        reward = 0
        if any(s in message for s in INTRINSICS_ACQUIRED_MSGS):
            reward = 1

        self.score += reward

        return reward




    
class PoisonResistanceScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        message = bytes(observation[env._message_index]).decode("latin-1").lower()

        reward = 0
        # ensure that we will give reward only once
        if self.score == 0:
            if "you feel healthy" in message or "you feel especially healthy" in message:
                self.score = 1
                reward = 1

        return reward

class SleepResistanceScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        message = bytes(observation[env._message_index]).decode("latin-1").lower()

        reward = 0
        # ensure that we will give reward only once
        if self.score == 0:
            if "you feel wide awake" in message:
                self.score = 1
                reward = 1

        return reward


class ColdResistanceScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        message = bytes(observation[env._message_index]).decode("latin-1").lower()

        reward = 0
        # ensure that we will give reward only once
        if self.score == 0:
            if "you feel full of hot air" in message:
                self.score = 1
                reward = 1

        return reward

class FireResistanceScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        message = bytes(observation[env._message_index]).decode("latin-1").lower()

        reward = 0
        # ensure that we will give reward only once
        if self.score == 0:
            if "you feel a momentary chill" in message:
                self.score = 1
                reward = 1

        return reward
    
    

    

class GoldScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        old_blstats = last_observation[env._blstats_index]
        blstats = observation[env._blstats_index]

        old_gold = old_blstats[nethack.NLE_BL_GOLD]
        gold = blstats[nethack.NLE_BL_GOLD]
        
        reward = gold - old_gold

        if info["end_status"] != self.StepStatus.RUNNING:
            reward = 0.0
        
        self.score += reward

        return reward



class EatingScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        old_internal = last_observation[env.unwrapped._internal_index]
        internal = observation[env.unwrapped._internal_index]

        reward = max(0, internal[7] - old_internal[7])
        self.score += reward

        return reward


class ExperienceScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        old_blstats = last_observation[env._blstats_index]
        blstats = observation[env._blstats_index]

        reward = float(blstats[nethack.NLE_BL_EXP] - old_blstats[nethack.NLE_BL_EXP])

        if info["end_status"] != self.StepStatus.RUNNING:
            reward = 0.0
        
        self.score += reward

        return reward    


class ScoutScore(Score):
    def __init__(self):
        super().__init__()
        self.dungeon_explored = {}

    def reward(self, env, last_observation, observation, info, training_info=None):
        glyphs = observation[env._glyph_index]
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        explored_old = 0
        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = explored - explored_old
        self.dungeon_explored[key] = explored
        self.score += reward

        return reward

    def reset_score(self):
        super().reset_score()
        self.dungeon_explored = {}


class StaircaseScore(Score):
    """
    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise.
    """

    def reward(self, env, last_observation, observation, info, training_info=None):
        internal = observation[env.unwrapped._internal_index]
        stairs_down = internal[4]

        reward = 1 if stairs_down else 0
        self.score += reward

        return reward


class StaircasePetScore(Score):
    """
    This task requires the agent to get on top of a staircase down (>), while
    having their pet next to it. See `NetHackStaircase` for the reward function.
    """

    def reward(self, env, last_observation, observation, info, training_info=None):
        internal = observation[env.unwrapped._internal_index]
        stairs_down = internal[4]

        reward = 0
        if stairs_down:
            glyphs = observation[env._glyph_index]
            blstats = observation[env._blstats_index]
            x, y = blstats[:2]

            neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]
            if np.any(nethack.glyph_is_pet(neighbors)):
                reward = 1

        self.score += reward

        return reward


class SokobanFillPitScore(Score):
    """
    This task requires the agent to put the boulders inside wholes for sokoban.
    We count each successful boulder moved into a whole as a total reward.
    """

    def reward(self, env, last_observation, observation, info, training_info=None):
        # the score counts how many pits we fill
        char_array = [chr(i) for i in observation[env._message_index]]
        message = "".join(char_array)

        if message.startswith("The boulder fills a pit.") or message.startswith(
            "The boulder falls into and plugs a hole in the floor!"
        ):
            reward = 1
        else:
            reward = 0
        self.score += reward

        return reward


class SokobanSolvedLevelsScore(Score):
    def __init__(self):
        super().__init__()
        self.sokoban_levels = {}

    def reward(self, env, last_observation, observation, info, training_info=None):
        glyphs = observation[env._glyph_index]
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        # TODO: this isn't error proof, one could create more holes / pits with a wand of digging
        # Note: we can't just check if the agent reached stairs because it can go out of the pits
        # this isn't "solving"
        if (dungeon_num, dungeon_level) == (4, 4):
            S_pit = nethack.GLYPH_CMAP_OFF + 52
            pits = np.isin(glyphs, [S_pit]).sum()
            key = (dungeon_num, dungeon_level)
            self.sokoban_levels[key] = pits
        elif (dungeon_num, dungeon_level) in [(4, 3), (4, 2), (4, 1)]:
            S_hole = nethack.GLYPH_CMAP_OFF + 54
            holes = np.isin(glyphs, [S_hole]).sum()
            key = (dungeon_num, dungeon_level)
            self.sokoban_levels[key] = holes

        self.score = 0
        for pits in self.sokoban_levels.values():
            # when all pits are filled we assume that sokoban level is solved
            if pits == 0:
                self.score += 1

    def reset_score(self):
        super().reset_score()
        self.sokoban_levels = {}


class SokobanReachedScore(Score):
    def __init__(self):
        super().__init__()
        self.reached_levels = {}

    def reward(self, env, last_observation, observation, info, training_info=None):
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        self.reached_levels[key] = 1

        self.score = 0
        for reached in self.reached_levels.keys():
            if reached in [(4, 4), (4, 3), (4, 2), (4, 1)]:
                self.score += 1

    def reset_score(self):
        super().reset_score()
        self.reached_levels = {}


class SokobanSolvedScore(Score):
    def __init__(self):
        super().__init__()

    def reward(self, env, last_observation, observation, info, training_info=None):
        glyphs = observation[env._glyph_index]
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        S_vcdoor = nethack.GLYPH_CMAP_OFF + 15  # /* closed door, vertical wall */
        S_hcdoor = nethack.GLYPH_CMAP_OFF + 16  # /* closed door, horizontal wall */
        door_closed = frozenset({S_vcdoor, S_hcdoor})
        if (dungeon_num, dungeon_level) == (4, 1) and np.isin(glyphs, [door_closed]).sum() == 0:
            self.score = 1

    def reset_score(self):
        super().reset_score()


class HealthScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        old_blstats = last_observation[env._blstats_index]
        blstats = observation[env._blstats_index]

        old_health = old_blstats[nethack.NLE_BL_HP]
        health = blstats[nethack.NLE_BL_HP]

        reward = health - old_health

        if info["end_status"] != self.StepStatus.RUNNING:
            reward = 0.0
        
        self.score += reward

        return reward

class ScoreScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        old_blstats = last_observation[env._blstats_index]
        blstats = observation[env._blstats_index]

        old_score = old_blstats[nethack.NLE_BL_SCORE]
        score = blstats[nethack.NLE_BL_SCORE]

        reward = score - old_score
        
        if info["end_status"] != self.StepStatus.RUNNING:
            reward = 0.0
        
        self.score += reward

        return reward

class KillsScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        message = bytes(observation[env._message_index]).decode("latin-1")

        if any(s in message.lower() for s in ('you kill', 'you destroy')):
            reward = 1
        else:
            reward = 0
        
        self.score += reward

        return reward


class EnhanceSkillScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        message = bytes(observation[env._message_index]).decode("latin-1").lower()

        if any(s in message for s in ('you are now more skilled', 'you are now most skilled')):
            reward = 1
        else:
            reward = 0
        
        self.score += reward

        return reward
    


class ProjectileScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        message = bytes(observation[env._message_index]).decode("latin-1")

        reward = 0

        
        if " hits the " in message:
            message = message.replace('!', '.')
            sentences = message.split('.')
            for sentence in sentences:
                if " hits the " in sentence:
                    splits = sentence.split(" hits the ")
                    sub = splits[0]
                    obj = splits[1]
                    sub = sub.split(' ')[-1]
                    if sub in PROJECTILES_SET:
                        obj = obj.replace('.', '')
                        if obj in MONSTERS_SET:
                            reward = 1
            

        '''
        if projectile_pattern.search(message) and not any(s in message for s in ('floor', 'ceiling', 'cavern', 'rock', 'falls', 'ground')):
            reward = 1
        else:
            reward = 0
        '''
        
        self.score += reward

        return reward
    


class ArmorScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        old_blstats = last_observation[env._blstats_index]
        blstats = observation[env._blstats_index]

        old_armor_class = old_blstats[nethack.NLE_BL_AC]
        armor_class = blstats[nethack.NLE_BL_AC]

        # remember, LOW armor class is better
        reward = - (armor_class - old_armor_class)

        if info["end_status"] != self.StepStatus.RUNNING:
            reward = 0.0
        
        self.score += reward

        return reward


class PickupFoodScore(Score):

    def _compute_non_corpse_comestibles(self, observation):
        inv_oclasses = observation[self._inv_oclasses_indx]
        comestible_indices = inv_oclasses == 7
        num_comestibles = np.sum(comestible_indices)
        if num_comestibles > 0:
            inv_strs = observation[self._inv_strs_indx]
            comestible_inv_strs = inv_strs[comestible_indices]
            num_corpses = np.sum([1 for item in comestible_inv_strs if b'corpse' in bytes(item)])
            num_comestibles -= num_corpses
        return num_comestibles
        
    
    def reward(self, env, last_observation, observation, info, training_info=None):

        self._inv_oclasses_indx = env._original_indices[env._original_observation_keys.index('inv_oclasses')]
        self._inv_strs_indx = env._original_indices[env._original_observation_keys.index('inv_strs')]

        last_num_comestibles = self._compute_non_corpse_comestibles(last_observation)
        num_comestibles = self._compute_non_corpse_comestibles(observation)

        reward = num_comestibles - last_num_comestibles

        if info["end_status"] != self.StepStatus.RUNNING:
            reward = 0.0
        
        self.score += reward

        return reward
    


class DlvlUpScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        old_blstats = last_observation[env._blstats_index]
        blstats = observation[env._blstats_index]

        old_dlvl = old_blstats[nethack.NLE_BL_DLEVEL]
        dlvl = blstats[nethack.NLE_BL_DLEVEL]

        reward = -(dlvl - old_dlvl)

        if info["end_status"] != self.StepStatus.RUNNING:
            reward = 0.0
        
        self.score += reward

        return reward

class DlvlDownScore(Score):
    def reward(self, env, last_observation, observation, info, training_info=None):
        old_blstats = last_observation[env._blstats_index]
        blstats = observation[env._blstats_index]

        old_dlvl = old_blstats[nethack.NLE_BL_DLEVEL]
        dlvl = blstats[nethack.NLE_BL_DLEVEL]

        reward = dlvl - old_dlvl

        if info["end_status"] != self.StepStatus.RUNNING:
            reward = 0.0
        
        self.score += reward

        return reward
    
    
    

class BucScore(Score):
    def __init__(self):
        super().__init__()

        self.patterns = {
            'buc': re.compile(r"(cursed|blessed|\(\d+:\d+\))")
        }
        
    def reward(self, env, last_observation, observation, info, training_info=None):
        msg = bytes(observation[env._message_index]).decode('latin-1')
        reward = 0.0
        # this uniquely defines messages where objects are identified by altar.
        # see: https://gist.github.com/tckmn/8078a34e3287ec32dadf#file-gistfile1-txt-L2849-L2850        
        if 'the altar.' in msg:
            if not bool(self.patterns['buc'].search(msg)):
                reward = 1.0

        self.score += reward

        return reward


class MessageScore(Score):
    def __init__(self, pos_msgs, neg_msgs, do_messages_expend):
        super().__init__()

        pos_messages = [msg for msg in pos_msgs.split(",") if msg]
        neg_messages = [msg for msg in neg_msgs.split(",") if msg]

        self.reward_messages = pos_messages + neg_messages
        self.messages_coeff = {k: 1. for k in pos_messages} | {k: -1. for k in neg_messages}
        self.messages_seen = {k: False for k in self.reward_messages}

        self.do_messages_expend = do_messages_expend


    def reward(self, env, last_observation, observation, info, training_info=None):
        msg = bytes(observation[env._message_index]).decode('latin-1')

        reward = 0

        for rmsg in self.reward_messages:
            if rmsg in msg and not self.messages_seen[rmsg]:
                reward += self.messages_coeff[rmsg]

                if self.do_messages_expend:
                    self.messages_seen[rmsg] = True

        self.score += reward

        return reward

    def reset_score(self):
        super().reset_score()

        self.messages_seen = {k: False for k in self.reward_messages}


class ExcaliburScore(Score):

    def reward(self, env, last_observation, observation, info, training_info=None):
        msg = bytes(observation[env._message_index]).decode('latin-1')
        reward = 0.0
        if "murky depths" in msg or "Excalibur" in msg:
            reward = 1.

        self.score += reward

        return reward

class ElberethScore(Score):

    def reward(self, env, last_observation, observation, info, training_info=None):
        msg = bytes(observation[env._message_index]).decode('latin-1')
        reward = 0.0
        if "elbereth" in msg.lower():
            reward = 1.

        self.score += reward

        return reward

class DenseElberethScore(Score):
    ELBERETH_ENGRAVE_MESSAGES = [
        "what do you want to write in the dust here? " + "elbereth"[:i] for i in range(len("elbereth")+1)
    ] + ['you read: "elbereth"']

    def __init__(self, reset_time):
        super().__init__()
        self.elbereth_progress = 0
        self.reset_time = reset_time
        self.timer = reset_time
        self.last_turns = 0

    def reward(self, env, last_observation, observation, info, training_info=None):
        msg = bytes(observation[env._message_index]).decode('latin-1')
        reward = 0.0

        if self.elbereth_progress < len(DenseElberethScore.ELBERETH_ENGRAVE_MESSAGES):
            next_engrave_message = DenseElberethScore.ELBERETH_ENGRAVE_MESSAGES[self.elbereth_progress]

            if next_engrave_message in msg.lower():
                # Big reward for finishing
                if self.elbereth_progress == len(DenseElberethScore.ELBERETH_ENGRAVE_MESSAGES)-1:
                    reward = 5.
                else:
                    reward = 1.
                self.elbereth_progress += 1
                self.timer = self.reset_time

            self.score += reward

        turns = info["episode_extra_stats"]["turns"]

        if self.timer <= 0:
            self.elbereth_progress = 0
            self.timer = self.reset_time
        # Only decrement timer if we've started writing
        elif self.elbereth_progress > 0:
            self.timer -= (turns - self.last_turns)
            self.last_turns = turns

        # print(self.elbereth_progress, reward, self.timer)

        return reward

    def reset_score(self):
        super().reset_score()

        self.elbereth_progress = 0
        self.timer = self.reset_time
        self.last_turns = 0
