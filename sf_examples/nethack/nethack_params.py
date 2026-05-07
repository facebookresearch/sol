from sample_factory.utils.utils import str2bool


def add_extra_params_nethack_env(parser):
    """
    Specify any additional command line arguments for NetHack environments.
    """
    p = parser
    p.add_argument(
        "--character", type=str, default="mon-hum-neu-mal", help="name of character. Defaults to 'mon-hum-neu-mal'."
    )
    p.add_argument(
        "--max_episode_steps",
        type=int,
        default=100000,
        help="maximum amount of steps allowed before the game is forcefully quit. In such cases, `info 'end_status']` will be equal to `StepStatus.ABORTED`",
    )
    p.add_argument(
        "--no_progress_timeout", type=int, default=30, help="after this many steps without acting in the env (incrementing a turn), exit."
    )    
    p.add_argument(
        "--penalty_step", type=float, default=0.0, help="constant applied to amount of frozen steps. Defaults to 0.0."
    )
    p.add_argument(
        "--penalty_time", type=float, default=0.0, help="constant applied to amount of frozen steps. Defaults to 0.0."
    )
    p.add_argument(
        "--fn_penalty_step",
        type=str,
        default="constant",
        help="name of the mode for calculating the time step penalty. Can be `constant`, `exp`, `square`, `linear`, or `always`. Defaults to `constant`.",
    )
    p.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="Path to save ttyrecs (game recordings) into, if save_ttyrec_every is nonzero. If nonempty string, interpreted as a path to a new or existing directory. If "
        " (empty string) or None, NLE choses a unique directory name.Defaults to `None`.",
    )
    p.add_argument(
        "--save_ttyrec_every",
        type=int,
        default=0,
        help="Integer, if 0, no ttyrecs (game recordings) will be saved. Otherwise, save a ttyrec every Nth episode.",
    )
    p.add_argument(
        "--use_prev_action",
        type=str2bool,
        default=True,
        help="If True, the model will use previous action. Defaults to `True`",
    )
    p.add_argument(
        "--use_prev_reward",
        type=str2bool,
        default=True,
        help="If True, the model will use previous reward. Defaults to `True`",
    )
    p.add_argument(
        "--use_elbereth_macro",
        type=str2bool,
        default=False,
        help="If True, ENGRAVE action always writes Elbereth with finger.",
    )
    p.add_argument(
        "--use_enhance_skills_macro",
        type=str2bool,
        default=False,
        help="If True, automatically ENHANCE skills when possible.",
    )
    p.add_argument(
        "--map_input_type",
        type=str,
        default="rgb",
        help="How to input the top-down map view. Current options are: rgb | glyphs. rgb renders to pixels, glyphs feeds in glyph indices.",
    )
    p.add_argument(
        "--inv_input_type",
        type=str,
        default="none",
        help="How to include the inventory as input.",
    )    
    p.add_argument("--crop_dim", type=int, default=18, help="Crop image around the player. Defaults to `18`.")
    p.add_argument(
        "--pixel_size",
        type=int,
        default=6,
        help="Rescales each character to size of `(pixel_size, pixel_size). Defaults to `6`.",
    )
    p.add_argument(
        "--use_inv_selection_wrapper",
        type=str2bool,
        default=False,
    )
    p.add_argument(
        "--use_menu_selection_wrapper",
        type=str2bool,
        default=False,
    )
    p.add_argument(
        "--use_spellcasting_wrapper",
        type=str2bool,
        default=False,
    )
    p.add_argument(
        "--max_spells",
        type=int,
        default=3,
    )
    p.add_argument(
        "--use_dungeon_overview_wrapper",
        type=str2bool,
        default=False,
    )
    p.add_argument(
        "--max_dungeon_overview_levels",
        type=int,
        default=15,
    )
    p.add_argument(
        "--use_attributes_wrapper",
        type=str2bool,
        default=False,
    )    
    p.add_argument(
        "--remove_writing_actions",
        type=str2bool,
        default=False,
    )



def add_extra_params_model(parser):
    """
    Specify any additional command line arguments for NetHack models.
    """
    p = parser
    p.add_argument(
        "--use_tty_only",
        type=str2bool,
        default=True,
        help="If True, the model will use tty_chars for the topline and bottomline. Defaults to `True`",
    )
    # parameters specific to SymbolicGlyphNet
    p.add_argument("--glyph_edim", type=int, default=64, help="Glyph Embedding Dim. Defaults to `64`")    
    p.add_argument(
        "--use_glyph_directions",
        type=str2bool,
        default=False,
        help="Add features showing directions between agent and certain glyphs",
    )    
    # parameters specific to ScaledNet
    p.add_argument("--h_dim", type=int, default=1738, help="Hidden dim for encoders. Defaults to `1738`")
    p.add_argument("--msg_hdim", type=int, default=64, help="Hidden dim for message encoder. Defaults to `64`")
    p.add_argument("--color_edim", type=int, default=16, help="Color Embedding Dim. Defaults to `16`")
    p.add_argument("--char_edim", type=int, default=16, help="Char Embedding Dim. Defaults to `16`")
    
    
    p.add_argument(
        "--use_crop",
        type=str2bool,
        default=True,
        help="Do we want to add additional embedding with cropped screen. Defaults to `True`",
    )
    p.add_argument(
        "--use_crop_norm",
        type=str2bool,
        default=True,
        help="Do we want to use BatchNorm2d when processing cropped screen. Defaults to `True`",
    )
    p.add_argument(
        "--screen_kernel_size",
        type=int,
        default=3,
        help="Kernel size for screen convolutional encoder. Defaults to `3`",
    )
    p.add_argument(
        "--no_max_pool", type=str2bool, default=False, help="Do we want to use max pool in ResNet. Defaults to `False`"
    )
    p.add_argument("--screen_conv_blocks", type=int, default=2, help="Number of blocks in ResNet. Defaults to `2`")
    p.add_argument("--blstats_hdim", type=int, default=512, help="Hidden dim for blstats encoder. Defaults to `512`")
    p.add_argument(
        "--fc_after_cnn_hdim", type=int, default=512, help="Hidden dim for screen encoder. Defaults to `512`"
    )
    p.add_argument(
        "--use_resnet",
        type=str2bool,
        default=False,
        help="Do we want to use ResNet in screen encoder. Defaults to `False`",
    )


def add_extra_params_general(parser):
    """
    Specify any additional command line arguments for NetHack.
    """
    p = parser
    p.add_argument(
        "--model", type=str, default="ChaoticDwarvenGPT5", help="Name of the model. Defaults to `ChaoticDwarvenGPT5`."
    )
    p.add_argument(
        "--add_stats_to_info",
        type=str2bool,
        default=True,
        help="If True, adds wrapper which loggs additional statisics. Defaults to `True`.",
    )
    p.add_argument(
        "--tokenizer_name", type=str, default="dnd", help="nle | dnd"
    )
    p.add_argument(
        "--max_token_length", type=int, default=12
    )

    p.add_argument(
        "--max_inv_items", type=int, default=24
    )

    p.add_argument(
        "--max_menu_items", type=int, default=10
    )

    p.add_argument(
        "--token_embed_std", type=float, default=0.01
    )

    p.add_argument(
        "--inv_encoder_type", type=str, default='att'
    )

    p.add_argument(
        "--inv_query_heads", type=int, default=4
    )

    p.add_argument(
        "--inv_query_std", type=float, default=0.01
    )
    


def add_extra_params_rewards(parser):
    """
    Specify params relating to intrinsic rewards and their coefficients.
    """
    p = parser
    p.add_argument(
        "--reward_scale_score", type=float, default=1.0, help="Score reward scale."
    )
    p.add_argument(
        "--reward_scale_scout", type=float, default=0.0, help="Scout reward scale."
    )
    p.add_argument(
        "--reward_scale_health", type=float, default=0.0, help="Health reward scale."
    )
    p.add_argument(
        "--reward_scale_gold", type=float, default=0.0, help="Gold reward scale."
    )
    p.add_argument(
        "--reward_scale_staircase", type=float, default=0.0, help="Stairs reward scale."
    )
    p.add_argument(
        "--reward_scale_eating", type=float, default=0.0, help="Eating reward scale."
    )
    p.add_argument(
        "--reward_scale_kills", type=float, default=0.0, help="Kills reward scale."
    )
    p.add_argument(
        "--reward_scale_intrinsics", type=float, default=0.0, help="Intrinsics reward scale."
    )
    p.add_argument(
        "--reward_scale_projectile", type=float, default=0.0, help="Projectile reward scale."
    )
    p.add_argument(
        "--reward_scale_enhance_skill", type=float, default=0.0, help="Enhance skill reward scale."
    )
    p.add_argument(
        "--reward_scale_armor", type=float, default=0.0, help="Armor reward scale."
    )
    p.add_argument(
        "--reward_scale_pickup_food", type=float, default=0.0, help="Pickup food reward scale."
    )
    p.add_argument(
        "--reward_scale_dlvl_up", type=float, default=0.0, help="Dlvl up reward scale."
    )
    p.add_argument(
        "--reward_scale_dlvl_down", type=float, default=0.0, help="Dlvl down reward scale."
    )
    p.add_argument(
        "--reward_scale_experience", type=float, default=0.0, help="Experience reward scale."
    )
    p.add_argument(
        "--reward_scale_message", type=float, default=0.0
    )
    p.add_argument(
        "--positive_reward_messages", type=str, default=""
    )
    p.add_argument(
        "--negative_reward_messages", type=str, default=""
    )
    p.add_argument(
        "--messages_expend", type=str2bool, default=True
    )
    p.add_argument(
        "--reward_scale_dense_elbereth", type=float, default=0.0
    )
    p.add_argument(
        "--reward_scale_buc", type=float, default=0.0
    )
    p.add_argument(
        "--reward_scale_sokoban_fill_pit", type=float, default=0.0
    )


def nethack_override_defaults(_env, parser):
    """RL params specific to NetHack envs."""
    # set hyperparameter values to the same as in d&d
    parser.set_defaults(
        use_record_episode_statistics=False,
        gamma=0.99,
        num_workers=48,
        num_envs_per_worker=32,
        worker_num_splits=2,
        train_for_env_steps=2_000_000_000,
        nonlinearity="relu",
        use_rnn=True,
        rnn_type="lstm",
        actor_critic_share_weights=True,
        policy_initialization="orthogonal",
        policy_init_gain=1.0,
        adaptive_stddev=False,  # True only for continous action distributions
        reward_scale=1.0,
        reward_clip=10000.0,
        batch_size=32768,
        rollout=256,
        max_grad_norm=4,
        num_epochs=1,
        num_batches_per_epoch=1,  # can be used for increasing the batch_size for SGD
        ppo_clip_ratio=0.1,
        ppo_clip_value=1.0,
        value_loss_coeff=0.5,
        exploration_loss="entropy",
        exploration_loss_coeff=0.003,
        learning_rate=0.0001,
        gae_lambda=1.0,
        with_vtrace=True,  # in d&d they've used vtrace
        normalize_input=False,  # turn off for now and use normalization from d&d
        normalize_returns=False,
        async_rl=True,
        experiment_summaries_interval=50,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-7,
        seed=22,
        save_every_sec=120,
        eval_env_frameskip=1,
    )
