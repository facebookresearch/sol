import math
import numpy as np
import gymnasium as gym

def remove_digits(s):
    return s
#    return ''.join([c for c in s if not c.isdigit()])

class HierarchicalWrapper(gym.Wrapper):

    def __init__(
            self,
            env,
            reward_scale,
            base_policies,
            controller_reward_key,
            num_policy_steps,
            controller_action_space_type,
            continuous_controller_min,
            continuous_controller_max,
            multi_num_coefs,
            multi_coef_spacing,
            multi_normalise_coefs,
            continuous_controller_min_for_task_reward=0,
            num_option_lengths=8,
            min_option_length=1,
    ):
        super().__init__(env)

        assert all(remove_digits(p) in reward_scale.keys() for p in base_policies), (
            f"base_policies ({base_policies}) must be in reward_scale dict: {reward_scale}"
        )

        assert controller_reward_key in reward_scale.keys(), (
            f"controller_reward_key ({controller_reward_key}) must be in reward_scale dict: {reward_scale}"
        )

        self.base_policies = base_policies
        self.policies = self.base_policies + ['controller']
        self.controller_reward_key = controller_reward_key
        self.controller_action_space_type = controller_action_space_type
        self.continuous_controller_min = continuous_controller_min
        self.continuous_controller_max = continuous_controller_max
        self.continuous_controller_min_for_task_reward = continuous_controller_min_for_task_reward
        self.multi_normalise_coefs = multi_normalise_coefs

        self.num_policy_steps = num_policy_steps
        self.reward_scale = reward_scale
        self.metrics = list(set(remove_digits(p) for p in base_policies))

        if self.controller_action_space_type == "discrete":
            controller_action_space = (gym.spaces.Discrete(len(self.base_policies)),)
        elif self.controller_action_space_type == "multidiscrete":
            controller_action_space = tuple(gym.spaces.Discrete(multi_num_coefs) for _ in range(len(self.base_policies)))

            if multi_coef_spacing == "linear":
                self.controller_multi_coefs = np.linspace(continuous_controller_min, continuous_controller_max, multi_num_coefs)
                if 'task_reward' in self.base_policies:
                    self.controller_multi_coefs_for_task_reward = np.linspace(continuous_controller_min_for_task_reward, continuous_controller_max, multi_num_coefs)
                
            elif multi_coef_spacing == "log":
                self.controller_multi_coefs = np.logspace(continuous_controller_min, continuous_controller_max, multi_num_coefs)
                # Transform from [1.0, 10.0] to [0.0, 1.0]
                self.controller_multi_coefs = (self.controller_multi_coefs - 1.0) / 9.0
            else:
                raise ValueError


        elif self.controller_action_space_type == "continuous":
            controller_action_space = (gym.spaces.Box(0, 1, shape=(len(self.base_policies),), dtype=np.float32),)
        else:
            raise ValueError


        # update the action space to now include an extra action representing the option chosen by the controller. 
        if not isinstance(self.action_space, gym.spaces.Tuple):
            self.action_space = gym.spaces.Tuple((self.action_space,) + controller_action_space)
            self._controller_action_space_index = 1
        else:
            self.action_space = gym.spaces.Tuple(self.env.action_space.spaces + controller_action_space)
            self._controller_action_space_index = len(self.env.action_space)

        
        if self.num_policy_steps == -1:
            # add a controller action representing option length
            self.action_space = gym.spaces.Tuple(self.action_space.spaces + (gym.spaces.Discrete(num_option_lengths),))

            self.option_execution_lengths = [2 ** (i + int(math.log2(min_option_length))) for i in range(num_option_lengths)]

            if self.controller_action_space_type in ["discrete", "continuous"]:
                self._controller_option_length_index = self._controller_action_space_index + 1
            elif self.controller_action_space_type == "multidiscrete":
                self._controller_option_length_index = self._controller_action_space_index + len(self.base_policies)
            else:
                raise ValueError

        self.action_space.controller_action_space_index = self._controller_action_space_index


        # update the observation space to include the option and controller rewards as well as policy indices
        obs_spaces = {
            "rewards": gym.spaces.Box(-math.inf, math.inf, shape=(len(self.policies),), dtype=np.float32),
            "current_policy_vec": gym.spaces.Box(0, 1, shape=(len(self.policies),), dtype=np.float32)
        }

        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)
        


    def _set_current_policy_to_controller(self):
        self.current_policy_vec = np.zeros(len(self.policies), dtype=np.float32)
        self.current_policy_vec[len(self.policies) - 1] = 1

    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()

        self.last_obs = obs.copy()
        self.last_info = info.copy()

        self._steps = 0

        # keep track of the returns for each policy
        self.policy_metrics = {}
        for p in self.policies:
            self.policy_metrics[p] = {m: 0.0 for m in self.metrics}

        # keep track of controller actions for logging
        self.controller_actions = []
        self.controller_option_lengths = []
        self.controller_reward = 0

        # first timestep uses controller policy
        self._set_current_policy_to_controller()
        obs["current_policy_vec"] = self.current_policy_vec
        obs['rewards'] = np.zeros(len(self.policies))

        return obs, info
    

    def step(self, action):

        low_level_action = action[:self._controller_action_space_index]

        if self.controller_action_space_type == "discrete":
            high_level_action_idx = action[self._controller_action_space_index]
            high_level_action_vec = np.zeros(len(self.base_policies), dtype=np.float32)
            high_level_action_vec[high_level_action_idx] = 1
        elif self.controller_action_space_type == "multidiscrete":
            high_level_action_tup = action[self._controller_action_space_index:self._controller_action_space_index+len(self.base_policies)]
            high_level_action_vec = self.controller_multi_coefs[np.array(high_level_action_tup)]
            if 'task_reward' in self.base_policies:
                task_reward_index = self.base_policies.index('task_reward')
                high_level_action_vec[task_reward_index] = self.controller_multi_coefs_for_task_reward[high_level_action_tup[task_reward_index]]

        elif self.controller_action_space_type == "continuous":
            high_level_action_vec = action[self._controller_action_space_index]

            def _sigmoid(x):
                return 1 / (1 + np.exp(-x))

            high_level_action_vec = _sigmoid(high_level_action_vec)
            high_level_action_vec = high_level_action_vec * (self.continuous_controller_max - self.continuous_controller_min) + self.continuous_controller_min
        else:
            raise ValueError

        if self.controller_action_space_type in ["multidiscrete", "continuous"]:
            if high_level_action_vec.sum() == 0:
                high_level_action_vec = np.ones(len(self.base_policies)) / len(self.base_policies)
            elif self.multi_normalise_coefs:
                high_level_action_vec /= high_level_action_vec.sum()

        # Controller
        if self.current_policy_vec[len(self.policies)-1] == 1:
            # switch the low-level policy, but otherwise no-op

            # we can't know the controller's reward yet, because it's in the future and depends on executing
            # the chosen option. Mark with NaN as a sentinel and backfill it in the learner thread.
            reward = - 0.42 #float('nan')

            # current policy selected by high-level action
            self.current_policy_vec = np.concatenate([high_level_action_vec, np.zeros(1, dtype=np.uint8)], axis=0)
            self.controller_actions.append(high_level_action_vec)

            # same as the last obs, but we change the policy index to reflect the chosen sub-policy
            observation = self.last_obs.copy()
            observation["current_policy_vec"] = self.current_policy_vec
                        
            observation['rewards'] = np.zeros(len(self.policies))
            observation['rewards'][self.policies.index('controller')] = reward


            if self.num_policy_steps == -1:
                self.current_option_length = self.option_execution_lengths[action[self._controller_option_length_index]]
            else:
                self.current_option_length = self.num_policy_steps

            self.controller_option_lengths.append(self.current_option_length)

            self._num_option_steps = 0
            
            return observation, reward, False, False, {}
        else:
            # step through regular env
            if not isinstance(low_level_action, int) and len(low_level_action) == 1:
                # the low-level action could consist of one number or two, if we are using the
                # inventory selection wrapper. NLE expects ints, so if it is a single number
                # we convert to int. 
                low_level_action = low_level_action[0]

            observation, task_reward, done, truncated, info = self.env.step(low_level_action)
            self.last_obs = observation.copy()
            self.last_info = info.copy()
            

            rewards = info["intrinsic_rewards"]
            rewards['task_reward'] = task_reward

            # log the returns for each policy and metric, and increment the current policy's returns
            for i, metric in enumerate(self.metrics):
                for j, policy in enumerate(self.base_policies):
                    # for SOL-discrete this accumulates the return for this metric whe
                    # for SOL-MD it does this whenever the coefficient is > 0, so it's more approximate
                    if self.current_policy_vec[j] != 0:
                        self.policy_metrics[policy][metric] += rewards[metric]

            reward = 0
            for policy_index, policy_name in enumerate(self.base_policies):
                policy_reward = rewards[remove_digits(policy_name)]
                policy_reward_scale = self.reward_scale[remove_digits(policy_name)]
                policy_indicator = self.current_policy_vec[policy_index]

                reward += policy_reward * policy_reward_scale * policy_indicator

                
            self._steps += 1
            self._num_option_steps += 1

            if self._num_option_steps == self.current_option_length:
                self._set_current_policy_to_controller()

            observation["current_policy_vec"] = self.current_policy_vec

            controller_reward = np.sum(
                [rewards[remove_digits(k)] * self.reward_scale[remove_digits(k)] for k in self.controller_reward_key.split('+')]
            )
            self.controller_reward += controller_reward


            if done or truncated:
                
                info['episode_extra_stats']['episode_controller_reward'] = self.controller_reward
                info['episode_extra_stats']['controller_option_length_mean'] = np.mean(self.controller_option_lengths)
                info['episode_extra_stats']['controller_option_length_std'] = np.std(self.controller_option_lengths)

                controller_reward_activations = np.stack(self.controller_actions) != 0
                
                for metric in self.metrics:
                    for i, policy in enumerate(self.base_policies):
                        info['episode_extra_stats'][f'{policy}_{metric}'] = self.policy_metrics[policy][metric] / (controller_reward_activations[:, i].sum() + 1e-6)
                        #info['episode_extra_stats'][f'{policy}_{metric}'] = self.policy_metrics[policy][metric] / (self.controller_actions.count(i) + 1e-6)
                

                logging_controller_actions = np.array(self.controller_actions)
                for i, policy in enumerate(self.base_policies):
                    info['episode_extra_stats'][f'{policy}_prob'] = logging_controller_actions[:, i].sum() / len(self.controller_actions)

                info['episode_extra_stats']['controller_coefficient_sum_mean'] = logging_controller_actions.sum(axis=-1).mean()

            # this controller reward will be accessed in the learner thread
            controller_reward *= self.reward_scale['controller']
                
            observation['rewards'] = np.zeros(len(self.policies))
            for policy in self.policies:
                if policy == 'controller':
                    observation['rewards'][self.policies.index('controller')] = controller_reward
                else:
                    observation['rewards'][self.policies.index(policy)] = np.sum([rewards[remove_digits(k)] * self.reward_scale[remove_digits(k)] for k in policy.split('+')])
                    
                
            self.rewards = rewards
            #print(reward, self.controller_reward)


            return observation, reward, done, truncated, info
