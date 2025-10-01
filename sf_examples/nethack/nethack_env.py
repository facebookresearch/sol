from typing import Optional

import gymnasium as gym

from sample_factory.utils.utils import is_module_available, log

# look for a patched NLE without lunar phases and such, otherwise use default
try:
    import nle_patched.nle as nle
    import nle_patched.nle.nethack as nethack
    log.info("Using patched NLE")
except ImportError:
    log.info("Could not find patched NLE, using default NLE instead")
    import nle
    import nle.nethack as nethack
    
from sf_examples.nethack.utils.wrappers import (
    BlstatsInfoWrapper,
    NoProgressTimeout,
    PrevActionsWrapper,
    MessageCountsWrapper, 
    TaskRewardsInfoWrapper,
    TileTTY,
)
from sf_examples.nethack.envs import (
    NetHackScoreFixedEat
)

from sol.hierarchical import HierarchicalWrapper
from sample_factory.algo.utils.flat_intrinsic_rewards import FlatIntrinsicRewardsWrapper

def nethack_available():
    return is_module_available("nle")


class NetHackSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


NETHACK_ENVS = [
    NetHackSpec("nethack_staircase", "NetHackStaircase-v0"),
    NetHackSpec("nethack_score", "NetHackScore-v0"),
    NetHackSpec("nethack_score_fixed_eat", "NetHackScoreFixedEat-v0"),
    NetHackSpec("nethack_pet", "NetHackStaircasePet-v0"),
    NetHackSpec("nethack_oracle", "NetHackOracle-v0"),
    NetHackSpec("nethack_gold", "NetHackGold-v0"),
    NetHackSpec("nethack_eat", "NetHackEat-v0"),
    NetHackSpec("nethack_scout", "NetHackScout-v0"),
    NetHackSpec("nethack_challenge", "NetHackChallenge-v0"),
    NetHackSpec("minihack_zombie_horde", "MiniHack-ZombieHorde-v0"),
    NetHackSpec("minihack_treasure_dash", "MiniHack-TreasureDash-v0"),
]


def nethack_env_by_name(name):
    for cfg in NETHACK_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown NetHack env")


def make_nethack_env(env_name, cfg, _env_config=None, render_mode: Optional[str] = None, **kwargs):
    nethack_spec = nethack_env_by_name(env_name)
    
    observation_keys = (
        "message",
        "blstats",
    )

    if cfg.map_input_type == "rgb":
        observation_keys += (
            "tty_chars",
            "tty_colors",
            "tty_cursor",
        )
    elif cfg.map_input_type == "glyphs":
        observation_keys += (
            "glyphs",
        )

    # ALSO AVAILABLE (OFF for speed)
    # "specials",
    # "colors",
    # "chars",
    # "glyphs",
    # "inv_glyphs",
    # "inv_strs",
    # "inv_letters",
    # "inv_oclasses",

    if render_mode is not None:
        if render_mode == "human" and "tty_chars" not in observation_keys:
            observation_keys += (
                "tty_chars",
                "tty_colors",
                "tty_cursor"
            )


    kwargs = dict(
        character=cfg.character,
        max_episode_steps=cfg.max_episode_steps,
        observation_keys=observation_keys,
        penalty_step=cfg.penalty_step,
        penalty_time=cfg.penalty_time,
        penalty_mode=cfg.fn_penalty_step,
        savedir=cfg.savedir,
        save_ttyrec_every=cfg.save_ttyrec_every,
        allow_all_yn_questions=True,
        #allow_all_modes=False,
    )

    #if env_name in ("nethack_staircase", "nethack_pet", "nethack_oracle"):
    #    kwargs.update(reward_win=cfg.reward_win, reward_lose=cfg.reward_lose)
            
    # else:  # print warning once
    # warnings.warn("Ignoring cfg.reward_win and cfg.reward_lose")

    env = gym.make(nethack_spec.env_id, render_mode=render_mode, **kwargs)

    env = NoProgressTimeout(env, no_progress_timeout=150)

    if cfg.map_input_type == "rgb":
        env = TileTTY(
            env,
            crop_size=cfg.crop_dim,
            rescale_font_size=(cfg.pixel_size, cfg.pixel_size),
        )

    if cfg.llm_reward_type in ["motif_legacy"]:
        env = MessageCountsWrapper(env)

    if cfg.use_prev_action:
        env = PrevActionsWrapper(env)

    if cfg.add_stats_to_info:
        env = BlstatsInfoWrapper(env)
        env = TaskRewardsInfoWrapper(env)

        

    if cfg.with_sol:
        reward_scale = {
            'score_score': cfg.reward_scale_score,
            'scout_score': cfg.reward_scale_scout,
            'health_score': cfg.reward_scale_health,
            'gold_score': cfg.reward_scale_gold,
            'staircase_score': cfg.reward_scale_staircase,
            'task_reward': 1.0,
            'controller': cfg.sol_controller_reward_scale,
        }

        # option rewards are in CSV format, i.e. "score,health", parse them here
        base_policies = [
            r + '_score' if 'task_reward' not in r else r for r in cfg.sol_option_rewards.split(',')
        ]
            
        controller_reward_key = cfg.sol_controller_reward_key
        if 'task_reward' not in controller_reward_key:
            controller_reward_key += '_score'
            
        env = HierarchicalWrapper(
            env,
            reward_scale,
            base_policies,
            controller_reward_key,
            cfg.sol_num_option_steps,
        )

    if cfg.with_flat_intrinsic_rewards:
        reward_scale = {
            'score_score': cfg.reward_scale_score,
            'scout_score': cfg.reward_scale_scout,
            'health_score': cfg.reward_scale_health,
            'gold_score': cfg.reward_scale_gold,
            'staircase_score': cfg.reward_scale_staircase,
        }

        env = FlatIntrinsicRewardsWrapper(env, reward_scale)
        
    return env

