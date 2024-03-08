import gymnasium as gym
import numpy as np

class RNDReward(gym.Wrapper):
    def __init__(self, env, rnd, scale_rate=None):
        super().__init__(env)
        self.rnd = rnd
        self.scale_rate = scale_rate
        self.mean = 0
        self.var = 1
        self.last_reward = None
    
    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        #NOTE: reward should be 0 since link env doesnt have a reward
        rnd_reward = self.rnd.step(obs)

        if self.scale_rate is not None:
            error = abs(rnd_reward - self.mean)
            scaled_rnd = (rnd_reward - self.mean) / self.var

            self.mean += (rnd_reward - self.mean) * self.scale_rate
            self.var += (error - self.var) * self.scale_rate

            #TODO: custom clip range?
            scaled_rnd = np.clip(scaled_rnd, -3, 3)
            rnd_reward = scaled_rnd
        
        self.last_reward = rnd_reward
        # print(rnd_reward)
        return obs, reward + rnd_reward, done, terminated, info