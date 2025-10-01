import numpy as np
import gymnasium as gym

class FlatIntrinsicRewardsWrapper(gym.Wrapper):

    def __init__(
            self,
            env,
            reward_scale,
    ):
        super().__init__(env)

        self.reward_scale = reward_scale

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self.total_orig_reward = 0
        return obs, info


    def step(self, action):
        observation, orig_reward, done, truncated, info = self.env.step(action)
        self.total_orig_reward += orig_reward
        
        intrinsic_rewards = info["intrinsic_rewards"]
        reward = np.sum(
            [intrinsic_rewards[k] * self.reward_scale[k] for k in self.reward_scale.keys()]
        )

        # track the original task reward whenever the episode ends
        if "episode_extra_stats" in info.keys():
            info["episode_extra_stats"]["total_orig_reward"] = self.total_orig_reward
            
        return observation, reward, done, truncated, info
