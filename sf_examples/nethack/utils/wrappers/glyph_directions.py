import gymnasium as gym
import numpy as np

from sf_examples.nethack.utils.text_to_glyphs import (
    NLE_TEXT_2_GLYPHS
)

class GlyphDirectionsWrapper(gym.Wrapper):
    def __init__(self, env, glyphs=['stairs down', 'stairs up']):
        super().__init__(env)
        self.glyphs = glyphs
        self.glyph_ids = [NLE_TEXT_2_GLYPHS[s] for s in self.glyphs]
        self.norm = np.array([1.0 / 79.0, 1.0 / 21.0])

        # Update observation space
        obs_spaces = {"glyph_directions": gym.spaces.Box(-1.0, 1.0, shape=(3 * len(self.glyphs),), dtype=np.float32)}
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = gym.spaces.Dict(obs_spaces)
        

    def _get_glyph_directions(self, obs):
        agent_pos = obs['blstats'][:2] 
        deltas = []
        for i in self.glyph_ids:
            glyph_pos = (obs['glyphs'] == i).nonzero()
            if len(glyph_pos[0]) == 0:
                glyph_pos = agent_pos
                found = 0.0
            else:
                glyph_pos = np.array((glyph_pos[1][:1], glyph_pos[0][:1])).squeeze()
                found = 1.0
            deltas.append(np.append((glyph_pos - agent_pos) * self.norm, found))
            
        return np.concatenate(deltas)



    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        deltas = self._get_glyph_directions(obs)
        obs['glyph_directions'] = deltas
        return obs, info


    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)
        deltas = self._get_glyph_directions(obs)
        obs['glyph_directions'] = deltas
        return obs, reward, term, trun, info
        

        
        
