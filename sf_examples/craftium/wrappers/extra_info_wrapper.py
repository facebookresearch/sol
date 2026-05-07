import gymnasium as gym
import numpy as np
import random


class ExtraInfoWrapper(gym.Wrapper):
    def __init__(
            self,
            env,
            compass_reward_directions,
            velocity_norm_clip=1000.0,
            velocity_penalty=1.0,
            darkness_penalty_threshold=0.0,
    ):
        super().__init__(env)

        obs_spaces = {
            "img": self.env.observation_space,
            "info_stats": gym.spaces.Box(-1000.0, 1000.0, shape=(9,), dtype=np.float32),
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # convert compass directions in degrees to (dx, dz) directions
        self.compass_reward_directions = {
            str(d): np.array((np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d))))
            for d in compass_reward_directions
        }

        self.velocity_norm_clip = velocity_norm_clip
        self.velocity_penalty = velocity_penalty
        self.darkness_penalty_threshold = darkness_penalty_threshold
        


    def _info_to_array(self, info):
        info_stats = np.concatenate([
            np.array(info["player_pos"], dtype=np.float32),
            np.array(info["player_vel"], dtype=np.float32),
            np.array((info["player_pitch"],), dtype=np.float32), 
            np.array((info["player_yaw"],), dtype=np.float32),
            np.array((np.log(1 + info["total_task_reward"] / 128),), dtype=np.float32)
        ])
        return info_stats

    
    def _clip_velocity(self, info):
        player_vel = info['player_vel']
        norm = np.linalg.norm(player_vel)
        if norm > self.velocity_norm_clip:
            player_vel = (player_vel / norm) * self.velocity_norm_clip
        info['player_vel'] = player_vel
        return info

        
    def _compute_direction_rewards(self, info):
        dx, dz = info['player_vel'][0], info['player_vel'][2]
        player_vel = np.array([dx, dz])
        direction_rewards = {}
        for name, direction in self.compass_reward_directions.items():
            direction_rewards[f'nav_{name}'] = np.dot(player_vel, direction).item()

        direction_rewards['up'] = info['player_vel'][1]
        direction_rewards['down'] = -info['player_vel'][1]
        direction_rewards['dig'] = -info['player_vel'][1] - self.velocity_penalty * np.linalg.norm(player_vel)

        return direction_rewards
        
                                    
    def reset(self, **kwargs):
        img_obs, info = self.env.reset(**kwargs)
        self.total_task_reward = 0
        info["total_task_reward"] = self.total_task_reward
        info = self._clip_velocity(info)
        info_stats = self._info_to_array(info)
        obs = {
            "img": img_obs,
            "info_stats": info_stats
        }
        
        self.last_inventory = info.get('inventory', {})

        return obs, info

    
    def step(self, action):
        img_obs, reward, term, trun, info = self.env.step(action)
        self.total_task_reward += reward
        
        info["total_task_reward"] = self.total_task_reward
        info = self._clip_velocity(info)
        info_stats = self._info_to_array(info)
            
        obs = {
            "img": img_obs,
            "info_stats": info_stats
        }

        intrinsic_rewards = {
            'task_reward': reward
        }
        
        direction_rewards = self._compute_direction_rewards(info)
        intrinsic_rewards.update(direction_rewards)

        brightness = np.mean(img_obs, axis=-1).mean()
        intrinsic_rewards['dark'] = -1 if brightness < self.darkness_penalty_threshold else 0  

        inventory = info.get('inventory', {})

        for key in ('gather_tree', 'gather_dirt', 'gather_stone', 'gather_iron', 'gather_diamond'):
            item_type = key.split('_')[1]
            num_items_old = sum(v for k,v in self.last_inventory.items() if (item_type in k and 'core' in k))
            num_items = sum(v for k,v in inventory.items() if (item_type in k and 'core' in k))
            intrinsic_rewards[key] = num_items - num_items_old

        
        info['episode_extra_stats'] = {}

        self.last_inventory = info.get('inventory', {})
        
        if term or trun:
            info['episode_extra_stats']['total_task_reward'] = self.total_task_reward


        info['intrinsic_rewards'] = intrinsic_rewards
        
        #print(info['intrinsic_rewards'])
        
        return obs, reward, term, trun, info
