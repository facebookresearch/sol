import platform
import sys
from typing import Optional

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

from sample_factory.utils.utils import is_module_available

from sample_factory.algo.utils.hierarchical import HierarchicalWrapper
from sample_factory.algo.utils.flat_intrinsic_rewards import FlatIntrinsicRewardsWrapper

from sf_examples.mujoco.option_rewards import (
    MazeTaskRewardsInfoWrapper,
    ExtraRewardsWrapper,
)
from sf_examples.mujoco.mazes import U_MAZE_FIXED, G_MAZE


def mujoco_available():
    # Disable on macOS Apple Silicon for now. TODO: fix the following:
    # gymnasium.error.DependencyNotInstalled: You are running an x86_64 build of Python on an Apple Silicon machine. This is not supported by MuJoCo. Please install and run a native, arm64 build of Python.. (HINT: you need to install mujoco, run `pip install gymnasium[mujoco]`.)
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return False
    return is_module_available("mujoco")


class MujocoSpec:
    def __init__(self, name, env_id, extra_args=None):
        self.name = name
        self.env_id = env_id
        self.extra_args = extra_args


MUJOCO_ENVS = [
    MujocoSpec("mujoco_hopper", "Hopper-v4"),
    MujocoSpec("mujoco_halfcheetah", "HalfCheetah-v4"),
    MujocoSpec("mujoco_humanoid", "Humanoid-v4"),
    MujocoSpec("mujoco_ant", "Ant-v4"),
    MujocoSpec("mujoco_standup", "HumanoidStandup-v4"),
    MujocoSpec("mujoco_doublependulum", "InvertedDoublePendulum-v4"),
    MujocoSpec("mujoco_pendulum", "InvertedPendulum-v4"),
    MujocoSpec("mujoco_reacher", "Reacher-v4"),
    MujocoSpec("mujoco_walker", "Walker2d-v4"),
    MujocoSpec("mujoco_pusher", "Pusher-v4"),
    MujocoSpec("mujoco_swimmer", "Swimmer-v4"),
    # PointMaze
    MujocoSpec("mujoco_point_umaze_sparse_cont", "PointMaze_UMaze-v3", {"continuing_task": True, "reward_type": "sparse"}),
    MujocoSpec("mujoco_point_umaze_dense_cont", "PointMaze_UMaze-v3", {"continuing_task": True, "reward_type": "dense"}),
    MujocoSpec("mujoco_point_umaze_sparse", "PointMaze_UMaze-v3", {"continuing_task": False, "reward_type": "sparse"}),
    MujocoSpec("mujoco_point_umaze_dense", "PointMaze_UMaze-v3", {"continuing_task": False, "reward_type": "dense"}),
    MujocoSpec("mujoco_point_gmaze_sparse", "PointMaze_UMaze-v3", {"continuing_task": False, "reward_type": "sparse", "maze_map": G_MAZE}),
    MujocoSpec("mujoco_point_gmaze_dense", "PointMaze_UMaze-v3", {"continuing_task": False, "reward_type": "dense", "maze_map": G_MAZE}),
]


def mujoco_env_by_name(name):
    for cfg in MUJOCO_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Mujoco env")


def make_mujoco_env(env_name, cfg, _env_config, render_mode: Optional[str] = 'rgb_array', **kwargs):
    mujoco_spec = mujoco_env_by_name(env_name)
    if mujoco_spec.extra_args is not None:
        env = gym.make(
            mujoco_spec.env_id,
            render_mode=render_mode,
            max_episode_steps=cfg.mujoco_max_episode_steps,
            **mujoco_spec.extra_args
        )
    else:
        env = gym.make(
            mujoco_spec.env_id,
            max_episode_steps=cfg.mujoco_max_episode_steps,
            render_mode=render_mode
        )

    if 'maze' in env_name.lower():
        env = MazeTaskRewardsInfoWrapper(
            env,
            reward_type=cfg.reward_type,
            success_reward_weight=cfg.mujoco_success_reward_weight
        )

    if cfg.with_flat_intrinsic_rewards:
        reward_scale = {
            'north': cfg.reward_scale_options,
            'south': cfg.reward_scale_options,
            'east': cfg.reward_scale_options,
            'west': cfg.reward_scale_options,
            'goal': cfg.reward_scale_goal, 
        }
        
        env = FlatIntrinsicRewardsWrapper(env, reward_scale)

        
    if cfg.with_sol:
        reward_scale = {
            'north': cfg.reward_scale_options,
            'south': cfg.reward_scale_options,
            'east': cfg.reward_scale_options,
            'west': cfg.reward_scale_options,
            'goal': cfg.reward_scale_goal,
            'controller': cfg.sol_controller_reward_scale,
        }
        base_policies = [r for r in cfg.sol_option_rewards.split(',')]
        controller_reward_key = 'goal'
        
        env = HierarchicalWrapper(
            env,
            reward_scale,
            base_policies,
            controller_reward_key,
            cfg.sol_num_option_steps,
        )

    env = ExtraRewardsWrapper(
        env,
        ctrl_cost_weight = cfg.mujoco_ctrl_cost_weight,
        contact_cost_weight = cfg.mujoco_contact_cost_weight,
        survive_cost_weight = cfg.mujoco_survive_cost_weight,
        terminate_when_unhealthy = cfg.mujoco_terminate_when_unhealthy,
    )
                
    return env
