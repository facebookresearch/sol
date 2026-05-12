import gymnasium as gym

from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface
from sf_examples.nethack.utils.task_rewards import (
    BucScore,
    EatingScore,
    ExperienceScore,
    GoldScore,
    ScoutScore,
    SokobanFillPitScore,
    SokobanReachedScore,
    StaircasePetScore,
    StaircaseScore,
    HealthScore,
    ScoreScore,
    KillsScore,
    ProjectileScore,
    ArmorScore,
    PickupFoodScore,
    EnhanceSkillScore,
    DlvlUpScore,
    DlvlDownScore,
    OracleScore,
    PoisonResistanceScore,
    SleepResistanceScore,
    ColdResistanceScore,
    FireResistanceScore,
    IntrinsicsScore,
    MessageScore, ExcaliburScore, ElberethScore, DenseElberethScore
)


class TaskRewardsInfoWrapper(gym.Wrapper, TrainingInfoInterface):
    def __init__(self, env: gym.Env, cfg):
        super().__init__(env)

        self.tasks = [
            EatingScore(),
            ExperienceScore(),
            GoldScore(),
            ScoutScore(),
            OracleScore(),
            #SokobanFillPitScore(),
            SokobanReachedScore(),
            StaircasePetScore(),
            StaircaseScore(),
            HealthScore(),
            ScoreScore(),
            KillsScore(),
            ProjectileScore(),
            ArmorScore(),
            PickupFoodScore(),
            EnhanceSkillScore(),
            DlvlUpScore(),
            DlvlDownScore(),
            BucScore(),
            PoisonResistanceScore(),
            SleepResistanceScore(),
            FireResistanceScore(),
            ColdResistanceScore(),
            IntrinsicsScore(),
            MessageScore(cfg.positive_reward_messages, cfg.negative_reward_messages, cfg.messages_expend),
            ExcaliburScore(),
            #ElberethScore(),
            #DenseElberethScore(cfg.dense_elbereth_reset_time),
        ]

        self.training_info = {}



    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        for task in self.tasks:
            task.reset_score()

        return obs

    def step(self, action):
        # use tuple and copy to avoid shallow copy (`last_observation` would be the same as `observation`)
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, reward, term, trun, info = self.env.step(action)
        observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        end_status = info["end_status"]

        # we will accumulate rewards for each step and log them when done signal appears
        # we also record individual rewards in case we want to use as intrinsic rewards
        timestep_rewards = {}
        for task in self.tasks:
           timestep_rewards[task.name] = task.reward(self.env.unwrapped, last_observation, observation, info,
                                                     training_info=self.training_info)

        if term or trun:
            info["episode_extra_stats"] = self.add_more_stats(info)


        info["intrinsic_rewards"] = timestep_rewards

        #print(info["intrinsic_rewards"])

        return obs, reward, term, trun, info

    def add_more_stats(self, info):
        extra_stats = info.get("episode_extra_stats", {})
        new_extra_stats = {task.name: task.score for task in self.tasks}
        return {**extra_stats, **new_extra_stats}
