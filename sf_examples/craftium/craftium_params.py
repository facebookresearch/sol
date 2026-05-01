from sample_factory.utils.utils import str2bool


def add_extra_params_craftium_env(parser):
    """
    Specify any additional command line arguments for Craftium environments.
    """
    p = parser
    p.add_argument(
        "--mt_port", type=int, default=49155, help="TCP port used by Minetest server and client communication."
    )
    p.add_argument(
        "--mt_wd", type=str, default="./", help="Directory where the Minetest working directories will be created."
    )
    p.add_argument(
        "--fps_max", type=int, default=2000, help="Max FPS."
    )
    p.add_argument(
        "--sync_mode", type=str2bool, default=True, help=""
    )
    p.add_argument(
        "--num_gpus", type=int, default=8, help=""
    )
    p.add_argument(
        "--num_compass_rewards", type=int, default=0, help=""
    )
    p.add_argument(
        "--reward_scale_compass_directions", type=float, default=0.0, help=""
    )
    p.add_argument(
        "--reward_scale_elevation", type=float, default=0.0, help=""
    )
    p.add_argument(
        "--reward_scale_gather", type=float, default=0.0, help=""
    )
    p.add_argument(
        "--reward_scale_dark", type=float, default=0.0, help=""
    )
    p.add_argument(
        "--darkness_penalty_threshold", type=float, default=30.0, help=""
    )
    p.add_argument(
        "--velocity_norm_clip", type=float, default=1000.0, help=""
    )
    p.add_argument(
        "--velocity_penalty", type=float, default=1.0, help=""
    )
    p.add_argument(
        "--sol_continuous_controller_min_for_task_reward", type=float, default=0.0, help=""
    )
    p.add_argument(
        "--use_clip_reward_wrapper", type=str2bool, default=False, help="Clip rewards to -1, 0, 1."
    )
    p.add_argument(
        "--max_episode_steps",
        type=int,
        default=10000,
        help="maximum amount of steps allowed before the game is forcefully quit.",
    )
    
