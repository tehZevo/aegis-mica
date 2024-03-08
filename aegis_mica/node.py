import os

import numpy as np
import yaml

from .link import Link

class Node:
    def load(path):
        with open(os.path.join(path, "config.yml"), "r") as f:
            config = yaml.safe_load(f.read())
        
        #TODO: default values if not exist
        node = Node(
            size=config["size"]
        )

        #load state if save data exists
        state_path = os.path.join(path, "state.npy")
        if os.path.exists(state_path):
            node.state = np.load(state_path)
        
        #create links
        for link_name, link_config in config["links"].items():
            link = Link(node=node, **link_config)

            #load link if save data exists
            link_path = os.path.join(path, "links", link_name)
            if os.path.exists(link_path):
                link.load(link_path)

            node.links[link_name] = link
        
        #start all links
        for link in node.links.values():
            link.start()

        return node
        
    def __init__(self, size):
        super().__init__()
        #TODO: other initializers?
        self.size = size
        self.state = np.zeros([size])
        self.links = {}
    
    def save(self, node_path):
        os.makedirs(node_path, exist_ok=True)
        
        #save config
        config = {
            "size": self.state.shape[0],
        }
        #add link configs
        link_configs = {name: link.get_config() for name, link in self.links.items()}
        config["links"] = link_configs
        
        with open(os.path.join(node_path, "config.yml"), "w") as f:
            f.write(yaml.dump(config))
        
        #save state
        np.save(os.path.join(node_path, "state"), self.state)

        #save links
        for link_name, link in self.links.items():
            link.save(os.path.join(node_path, "links", link_name))

    def update(self):
        #dont update state if we have no links
        if len(self.links) == 0:
            #TODO: remove print
            print("nothing to update...")
            return
            
        #wait for all of our links
        for link in self.links.values():
            link.env.action_updated.wait()
            link.env.action_updated.clear()

        actions = [link.env.last_action for link in self.links.values()]
        #TODO: other methods of reduce?
        self.state = np.mean(actions, axis=0)

        #TODO: if save issues occur, split the continue_step.set()s into a graph-triggered function
        
        #step envs
        for link in self.links.values():
            link.env.continue_step.set()

        rnd_rewards = [link.rnd_env.last_reward for link in self.links.values()]
        if any([r is None for r in rnd_rewards]):
            return 0
            
        mean_reward = np.mean(rnd_rewards)

        return mean_reward
            
    def add_link(self, name, link):
        if name in self.links:
            raise ValueError(f"Link '{name}' already exists")

        self.links[name] = link
    
    def remove_link(self, name):
        if name not in self.links:
            raise ValueError(f"Link '{name}' does not exist")
            
        link = self.links[name]
        del self.links[name]
        
        return link
        
        
