from threading import Event
import time

import gymnasium as gym
from protopost import protopost_client as ppcl
from requests.exceptions import ConnectionError
from nd_to_json import json_to_nd

class LinkEnv(gym.Env):
    def __init__(self, source_url, source_size, node_size):
        super().__init__()
        self.source_url = source_url
        self.source_size = source_size
        self.node_size = node_size
        self.last_action = None
        self.action_updated = Event()
        self.continue_step = Event()

        #TODO: customizable range for observation?
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[self.source_size])
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=[self.node_size])
    
    def get_obs(self):
        while True:
            try:
                return json_to_nd(ppcl(self.source_url))
            except ConnectionError as e:
                print(f"Failed to get observation from {self.source_url}, retrying...")
                #TODO: configurable fail sleep
                time.sleep(1)
        
    def step(self, action):
        self.last_action = action

        #tell node we're ready
        self.action_updated.set()

        #wait here for the node to unlock us
        self.continue_step.wait()
        self.continue_step.clear()
        
        obs = self.get_obs()

        return obs, 0, False, False, {}
    
    def reset(self, **kwargs):
        obs = self.get_obs()
        return obs, {}