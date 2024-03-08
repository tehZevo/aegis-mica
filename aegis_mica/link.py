from threading import Thread
import traceback
from uuid import uuid4
import os

import gymnasium as gym
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from rnd import RND
import yaml
import numpy as np

from .link_env import LinkEnv
from .wrappers import ConcatNodeState, RNDReward

class Link:
    def __init__(self, node, url, size, recurrent=True, algorithm="ppo"):
        self.node = node
        self.url = url
        self.size = size
        self.recurrent = recurrent
        self.algorithm = algorithm.strip().lower()

        self.setup_env()
        self.setup_agent()

    def setup_env(self):
        self.env = LinkEnv(self.url, self.size, self.node.size)
        #TODO: optional curiousity?
        #TODO: customizable RND learning rate
        self.rnd = RND(self.size)
        wrapped_env = self.env
        #TODO: parameters for scale and clip
        wrapped_env = RNDReward(wrapped_env, self.rnd, scale_rate=1e-3)
        self.rnd_env = wrapped_env
        
        if self.recurrent:
            wrapped_env = ConcatNodeState(wrapped_env, self.node)

        if self.algorithm == "ppo":
            wrapped_env = DummyVecEnv([lambda: wrapped_env])
        
        self.wrapped_env = wrapped_env
        
    
    def setup_agent(self):
        if self.algorithm == "ppo":
            #TODO: configurable LR and other ppo params?
            self.agent = PPO("MlpPolicy", self.wrapped_env, verbose=0, n_steps=256, batch_size=32, n_epochs=4)
        elif self.algorithm == "ddpg":
            #TODO: params for action noise and LR?
            n_actions = wrapped_env.action_space.shape[0]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            self.agent = DDPG("MlpPolicy", self.wrapped_env, verbose=0, action_noise=action_noise)
        else:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'")
        
    def load(self, link_path):
        if self.algorithm == "ppo":
            self.agent = PPO.load(os.path.join(link_path, "agent"), env=self.wrapped_env)
        elif self.algorithm == "ddpg":
            self.agent = DDPG.load(os.path.join(link_path, "agent"), env=self.wrapped_env)
        else:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'")
        
        self.rnd.load(os.path.join(link_path, "rnd"))
    
    #data to store in node's config
    def get_config(self):
        return {
            "url": self.url,
            "size": self.size,
            "recurrent": self.recurrent,
            "algorithm": self.algorithm,
        }

    def save(self, link_path):
        os.makedirs(link_path, exist_ok=True)
        self.agent.save(os.path.join(link_path, "agent"))
        self.rnd.save(os.path.join(link_path, "rnd"))
        
    def start(self):
        def run():
            try:
                self.agent.learn(total_timesteps=float("inf"))
            except Exception as e:
                print(traceback.format_exc())

        #TODO: store thread in links dict?
        #TODO: method to stop thread so we can add/remove links at runtime
        self.thread = Thread(target=run, daemon=True)
        self.thread.start()