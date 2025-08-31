import gymnasium as gym
import numpy as np



class MazeTaskRewardsInfoWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            reward_type='default',
            ctrl_cost_weight=0.0,
            contact_cost_weight=0.0,
            survive_cost_weight=0.0,
            success_reward_weight=0.0, 
            terminate_when_unhealthy=False,
    ):
        self.reward_type = reward_type
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.survive_cost_weight = survive_cost_weight
        self.success_reward_weight = success_reward_weight
        self.terminate_when_unhealthy = terminate_when_unhealthy
        super().__init__(env)
        

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs.copy()
        self.total_x_velocity = 0
        self.total_y_velocity = 0

        return obs, info

    def _distance_to_goal(self, obs):
        return np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])

    def step(self, action):
                
        obs, reward, term, trun, info = self.env.step(action)
        
        last_x, last_y = self.last_obs['achieved_goal']
        current_x, current_y = obs['achieved_goal']
        
        dx = current_x - last_x
        dy = current_y - last_y

        if self.reward_type == 'diff':
            reward = self._distance_to_goal(self.last_obs) - self._distance_to_goal(obs)
        elif self.reward_type == 'sparse':
            reward = 0.0
        else:
            assert self.reward_type == 'default'

        reward += self.success_reward_weight * float(info['success'])

        self.total_x_velocity += dx
        self.total_y_velocity += dy


        extra_reward = 0

        
        if 'reward_ctrl' in info.keys():
            extra_reward += self.ctrl_cost_weight * info['reward_ctrl']
            
        if 'reward_contact' in info.keys():
            extra_reward += self.contact_cost_weight * info['reward_contact']
            
        if 'reward_survive' in info.keys():
            #extra_reward += self.survive_cost_weight * info['reward_survive']
            extra_reward -= self.survive_cost_weight * (1 - info['reward_survive'])
            
        

        # for some reason, passing this argument the regular way in kwargs doesn't work
        # so we do it manually
        if  self.terminate_when_unhealthy and info['reward_survive'] != 1:
            term = True
        

        if term or trun:
            info["episode_extra_stats"] = self.add_more_stats(info)
                
        intrinsic_rewards = {
            'north': dy,
            'south': -dy,
            'east': dx,
            'west': -dx,
            'goal': reward,
        }
            
        intrinsic_rewards = {key: val + extra_reward for key, val in intrinsic_rewards.items()}
        info['intrinsic_rewards'] = intrinsic_rewards

        self.last_obs = obs.copy()
        return obs, reward, term, trun, info

    
    def add_more_stats(self, info):
        extra_stats = info.get("episode_extra_stats", {})
        new_extra_stats = {'success': int(info['success'])}
        new_extra_stats['total_x_velocity'] = self.total_x_velocity
        new_extra_stats['total_y_velocity'] = self.total_y_velocity
        #new_extra_stats['final_goal_reward'] = info['intrinsic_rewards']['goal']
        return {**extra_stats, **new_extra_stats}
    
        




class ExtraRewardsWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            ctrl_cost_weight=0.0,
            contact_cost_weight=0.0,
            survive_cost_weight=0.0,
            terminate_when_unhealthy=True, 
    ):
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.survive_cost_weight = survive_cost_weight
        self.terminate_when_unhealthy = terminate_when_unhealthy
        super().__init__(env)
        

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.total_reward_ctrl = 0
        self.total_reward_contact = 0
        self.total_reward_survive = 0
        return obs, info


    def step(self, action):
                
        obs, reward, term, trun, info = self.env.step(action)

        
        extra_reward = 0
        
        if 'reward_ctrl' in info.keys():
            extra_reward += self.ctrl_cost_weight * info['reward_ctrl']
            self.total_reward_ctrl += info['reward_ctrl']
            
        if 'reward_contact' in info.keys():
            extra_reward += self.contact_cost_weight * info['reward_contact']
            self.total_reward_contact += info['reward_contact']
            
        if 'reward_survive' in info.keys():
            #extra_reward += self.survive_cost_weight * info['reward_survive']
            extra_reward -= self.survive_cost_weight * (1 - info['reward_survive'])
            self.total_reward_survive += info['reward_survive']

        
            # for some reason, passing this argument the regular way in kwargs doesn't work
            # so we do it manually
            if  self.terminate_when_unhealthy and info['reward_survive'] != 1:
                term = True
        

        if term or trun:
            info["episode_extra_stats"] = self.add_more_stats(info)

        reward = reward + extra_reward
                
        return obs, reward, term, trun, info

    
    def add_more_stats(self, info):
        extra_stats = info.get("episode_extra_stats", {})
        new_extra_stats = {
            'total_reward_ctrl': self.total_reward_ctrl,
            'total_reward_contact': self.total_reward_contact,
            'total_reward_survive': self.total_reward_survive,
        }
        return {**extra_stats, **new_extra_stats}
    
    
