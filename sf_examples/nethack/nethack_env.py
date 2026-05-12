from typing import Optional

import enum
import gymnasium as gym
import nle_patched.nle as nle  # noqa: F401
import nle_patched.nle.nethack as nethack

from sample_factory.utils.utils import is_module_available
from sf_examples.nethack.utils.wrappers import (
    BlstatsInfoWrapper,
    NoProgressTimeout,
    PrevActionsWrapper,
    PrevRewardsWrapper,
    MessageCountsWrapper,
    TaskRewardsInfoWrapper,
    MenuSelectionWrapper,
    NLETokenizerWrapper,
    GlyphDirectionsWrapper,
    ExtraObsWrapper,
    DungeonOverviewWrapper,
    ElberethWrapper,
    EnhanceSkillsWrapper,
    TileTTY,
)
from sf_examples.nethack.envs import (
    NetHackScoreFixedEat
)

from sample_factory.algo.utils.hierarchical import HierarchicalWrapper
from sample_factory.algo.utils.flat_intrinsic_rewards import FlatIntrinsicRewardsWrapper
from sf_examples.nethack.utils.wrappers.extra_info import ExtraInfoWrapper


def nethack_available():
    return is_module_available("nle")


class NetHackSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


NETHACK_ENVS = [
    # classic envs from NLE paper
    NetHackSpec("nethack_staircase", "NetHackStaircase-v0"),
    NetHackSpec("nethack_score", "NetHackScore-v0"),
    NetHackSpec("nethack_score_fixed_eat", "NetHackScoreFixedEat-v0"),
    NetHackSpec("nethack_pet", "NetHackStaircasePet-v0"),
    NetHackSpec("nethack_oracle", "NetHackOracle-v0"),
    NetHackSpec("nethack_gold", "NetHackGold-v0"),
    NetHackSpec("nethack_eat", "NetHackEat-v0"),
    NetHackSpec("nethack_scout", "NetHackScout-v0"),
    # challenge
    NetHackSpec("nethack_challenge", "NetHackScore-v0"),
    # minihack envs from SOL paper
    NetHackSpec("minihack_gem_stack", "MiniHack-GemStack-v0"),
    NetHackSpec("minihack_zombie_horde", "MiniHack-ZombieHorde-v0"),
    NetHackSpec("minihack_treasure_dash", "MiniHack-TreasureDash-v0"),
    # diagnostic envs
    NetHackSpec("minihack_ring_inv", "MiniHack-RingInv-v0"),
    NetHackSpec("minihack_armor_inv", "MiniHack-ArmorInv-v0"),
    NetHackSpec("minihack_armor_uncursed", "MiniHack-ArmorUncursed-v0"),
    NetHackSpec("minihack_armor_identify", "MiniHack-ArmorIdentify-v0"),
    NetHackSpec("minihack_armor_enchant", "MiniHack-ArmorEnchant-v0"),
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

    if cfg.use_menu_selection_wrapper or cfg.inv_input_type == "tok":
        observation_keys += (
            "inv_strs",
            "inv_letters",
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


    if cfg.use_dungeon_overview_wrapper or cfg.use_menu_selection_wrapper:
        observation_keys += (
            "tty_chars",
        )

    observation_keys += ("inv_oclasses",)


    if render_mode is not None:
        if render_mode == "human":
            for key in ("tty_chars", "tty_colors", "tty_cursor"):
                if key not in observation_keys:
                    observation_keys += (key,)



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

    if env_name == "nethack_challenge":
        actions = list(nethack.ACTIONS)

        kwargs.update(
            actions=tuple(actions),
            allow_all_modes=True,
        )

    #if env_name in ("nethack_staircase", "nethack_pet", "nethack_oracle"):
    #    kwargs.update(reward_win=cfg.reward_win, reward_lose=cfg.reward_lose)

    # else:  # print warning once
    # warnings.warn("Ignoring cfg.reward_win and cfg.reward_lose")

    env = gym.make(nethack_spec.env_id, render_mode=render_mode, **kwargs)

    env = NoProgressTimeout(env, no_progress_timeout=cfg.no_progress_timeout)

    if cfg.map_input_type == "rgb":
        env = TileTTY(
            env,
            crop_size=cfg.crop_dim,
            rescale_font_size=(cfg.pixel_size, cfg.pixel_size),
        )

    if cfg.llm_reward_type in ["motif", "motif_legacy"]:
        env = MessageCountsWrapper(env)

    if cfg.use_prev_action:
        env = PrevActionsWrapper(env)

    if cfg.use_prev_reward:
        env = PrevRewardsWrapper(env)

    if cfg.use_glyph_directions:
        env = GlyphDirectionsWrapper(
            env,
            glyphs=["doorway", "vertical closed door", "horizontal closed door"],
        )

    if cfg.add_stats_to_info:
        env = BlstatsInfoWrapper(env)
        env = TaskRewardsInfoWrapper(env, cfg)
        env = ExtraInfoWrapper(env)

    if cfg.use_dungeon_overview_wrapper:
        env = DungeonOverviewWrapper(
            env,
            max_rows = cfg.max_dungeon_overview_levels
        )

    if cfg.use_attributes_wrapper:
        env = ExtraObsWrapper(env)

    if cfg.use_menu_selection_wrapper:
        env = MenuSelectionWrapper(env, do_gui_menu_selection=True)

        env = NLETokenizerWrapper(
            env,
            cfg.tokenizer_name,
            cfg.max_token_length,
            cfg.max_inv_items,
            cfg.max_menu_items,
        )

    reward_scale = {
        'score_score': cfg.reward_scale_score,
        'scout_score': cfg.reward_scale_scout,
        'health_score': cfg.reward_scale_health,
        'gold_score': cfg.reward_scale_gold,
        'experience_score': cfg.reward_scale_experience,
        'dlvl_up_score': cfg.reward_scale_dlvl_up,
        'dlvl_down_score': cfg.reward_scale_dlvl_down,
        'staircase_score': cfg.reward_scale_staircase,
        'eating_score': cfg.reward_scale_eating,
        'pickup_food_score': cfg.reward_scale_pickup_food,
        'kills_score': cfg.reward_scale_kills,
        'projectile_score': cfg.reward_scale_projectile,
        'enhance_skill_score': cfg.reward_scale_enhance_skill,
        'armor_score': cfg.reward_scale_armor,
        'intrinsics_score': cfg.reward_scale_intrinsics,
        'buc_score': cfg.reward_scale_buc,
    }


    if cfg.with_sol:
        sol_reward_scale = reward_scale
        sol_reward_scale["task_reward"] = 1.0
        sol_reward_scale["controller"] = cfg.sol_controller_reward_scale

        base_policies = [
            r + '_score' if 'task_reward' not in r else r for r in cfg.sol_option_rewards.split(',')
        ]

        controller_reward_key = cfg.sol_controller_reward_key

        if 'task_reward' not in controller_reward_key:
            controller_reward_key += '_score'

        env = HierarchicalWrapper(
            env,
            sol_reward_scale,
            base_policies,
            controller_reward_key,
            cfg.sol_num_option_steps,
            cfg.sol_controller_action_space,
            cfg.sol_continuous_controller_min,
            cfg.sol_continuous_controller_max,
            cfg.sol_multi_num_coefs,
            cfg.sol_multi_coef_spacing,
            cfg.sol_multi_normalise_coefs
        )

    if cfg.with_flat_intrinsic_rewards:
        env = FlatIntrinsicRewardsWrapper(env, reward_scale)


    return env

