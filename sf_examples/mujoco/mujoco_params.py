from sample_factory.utils.utils import str2bool

def mujoco_override_defaults(env, parser):
    parser.set_defaults(
        batched_sampling=False,
        num_workers=8,
        num_envs_per_worker=8,
        worker_num_splits=2,
        train_for_env_steps=10000000,
        encoder_mlp_layers=[64, 64],
        env_frameskip=1,
        nonlinearity="tanh",
        batch_size=1024,
        kl_loss_coeff=0.1,
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        reward_scale=1,
        rollout=64,
        max_grad_norm=3.5,
        num_epochs=2,
        num_batches_per_epoch=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=1.3,
        exploration_loss_coeff=0.0,
        learning_rate=0.00295,
        lr_schedule="linear_decay",
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        normalize_input=True,
        normalize_returns=True,
        value_bootstrap=True,
        experiment_summaries_interval=5,
        save_every_sec=120,
        serial_mode=False,
        async_rl=False,
    )


# noinspection PyUnusedLocal
def add_mujoco_env_args(env, parser):
    p = parser
    p.add_argument(
        "--reward_type", type=str, default='default', help="default | diff | x_velocity"
    )
    p.add_argument(
        "--mujoco_max_episode_steps", type=int, default=500, help="max episode steps"
    )
    p.add_argument(
        "--mujoco_ctrl_cost_weight", type=float, default=0.0, help="control cost"
    )
    p.add_argument(
        "--mujoco_contact_cost_weight", type=float, default=0.0, help="contact cost"
    )
    p.add_argument(
        "--mujoco_survive_cost_weight", type=float, default=0.0, help="survival reward"
    )
    p.add_argument(
        "--mujoco_success_reward_weight", type=float, default=0.0, help="success reward"
    )
    p.add_argument(
        "--mujoco_terminate_when_unhealthy", type=str2bool, default=False, help="end episode when unhealth (e.g. upside down)"
    )
    p.add_argument(
        "--reward_scale_options", type=float, default=1.0, help="scaling of NSEW"
    )
    p.add_argument(
        "--reward_scale_goal", type=float, default=1.0, help="scaling of goal reaching rewards"
    )

