import math
import numpy as np
import gymnasium as gym

def remove_digits(s):
    return ''.join([c for c in s if not c.isdigit()])

class HierarchicalWrapper(gym.Wrapper):

    def __init__(
            self,
            env,
            reward_scale,
            base_policies,
            controller_reward_key,
            num_policy_steps,
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

        self.num_policy_steps = num_policy_steps
        self.reward_scale = reward_scale        
        self.metrics = list(set(remove_digits(p) for p in base_policies))


        # update the action space to now include an extra action representing the option chosen by the controller. 
        if not isinstance(self.action_space, gym.spaces.Tuple):
            self.action_space = gym.spaces.Tuple((self.action_space, gym.spaces.Discrete(len(self.base_policies))))
            self._controller_action_space_index = 1
        else:
            self.action_space = gym.spaces.Tuple(self.env.action_space.spaces + (gym.spaces.Discrete(len(self.base_policies)),))
            self._controller_action_space_index = len(self.env.action_space)

        
        if self.num_policy_steps == -1:
            # add a controller action representing option length
            num_policy_steps_choices = 8
            self.action_space = gym.spaces.Tuple(self.action_space.spaces + (gym.spaces.Discrete(num_policy_steps_choices),))
            self._controller_option_length_index = self._controller_action_space_index + 1

            
            

        # update the observation space to include the option and controller rewards as well as policy indices
        obs_spaces = {
            "rewards": gym.spaces.Box(-math.inf, math.inf, shape=(len(self.policies),), dtype=np.float32),
            "current_policy": gym.spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
        }
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)
        



    
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
        self.current_policy = 'controller'
            
        obs['current_policy'] = np.array([self.policies.index(self.current_policy)], dtype=np.uint8)
        obs['rewards'] = np.zeros(len(self.policies))


        return obs, info
    

    def step(self, action):

        low_level_action = action[:self._controller_action_space_index]
        high_level_action = action[self._controller_action_space_index]

        if isinstance(high_level_action, np.ndarray):
            assert len(high_level_action) == 1
            high_level_action = high_level_action[0]

        if isinstance(low_level_action, list):
            assert len(low_level_action) == 1
            low_level_action = low_level_action[0]


        
        if self.current_policy == 'controller':
            # switch the low-level policy, but otherwise no-op

            # we can't know the controller's reward yet, because it's in the future and depends on executing
            # the chosen option. So mark it and compute it later in the learner thread. 
            reward = -0.42

            # current policy selected by high-level action
            self.current_policy = self.base_policies[high_level_action]
            self.controller_actions.append(high_level_action)

            # same as the last obs, but we change the policy index to reflect the chosen sub-policy
            observation = self.last_obs.copy()
            observation['current_policy'] = np.array([self.policies.index(self.current_policy)], dtype=np.uint8)
                        
            observation['rewards'] = np.zeros(len(self.policies))
            observation['rewards'][self.policies.index('controller')] = reward


            if self.num_policy_steps == -1:
                self.current_option_length = 2 ** action[self._controller_option_length_index]
            else:
                self.current_option_length = self.num_policy_steps

            self.controller_option_lengths.append(self.current_option_length)

            self._num_option_steps = 0
            
            return observation, reward, False, False, {}
        else:
            # step through regular env
            if len(low_level_action) == 1:
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
            for metric in self.metrics:
                self.policy_metrics[self.current_policy][metric] += rewards[metric]

                    
            reward = np.sum(
                [rewards[remove_digits(k)] * self.reward_scale[remove_digits(k)] for k in self.current_policy.split('+')]
            )
                    
            self._steps += 1
            self._num_option_steps += 1

            if self._num_option_steps == self.current_option_length:
                self.current_policy = 'controller'

            observation['current_policy'] = np.array([self.policies.index(self.current_policy)], dtype=np.uint8)

            controller_reward = np.sum(
                [rewards[remove_digits(k)] * self.reward_scale[remove_digits(k)] for k in self.controller_reward_key.split('+')]
            )
            self.controller_reward += controller_reward


            if done or truncated:
                
                info['episode_extra_stats']['episode_controller_reward'] = self.controller_reward
                info['episode_extra_stats']['controller_option_length_mean'] = np.mean(self.controller_option_lengths)
                info['episode_extra_stats']['controller_option_length_std'] = np.std(self.controller_option_lengths)
                
                for metric in self.metrics:
                    for i, policy in enumerate(self.base_policies):
                        info['episode_extra_stats'][f'{policy}_{metric}'] = self.policy_metrics[policy][metric] / (self.controller_actions.count(i) + 1e-6)

                for i, policy in enumerate(self.base_policies):
                    info['episode_extra_stats'][f'{policy}_prob'] = self.controller_actions.count(i) / len(self.controller_actions)

            
            controller_reward *= self.reward_scale['controller']
                
            observation['rewards'] = np.zeros(len(self.policies))
            for policy in self.policies:
                if policy == 'controller':
                    observation['rewards'][self.policies.index('controller')] = controller_reward
                else:
                    observation['rewards'][self.policies.index(policy)] = np.sum([rewards[remove_digits(k)] * self.reward_scale[remove_digits(k)] for k in policy.split('+')])
                    
                
            self.rewards = rewards
            #print(rewards, self.episode_controller_reward)
            return observation, reward, done, truncated, info
