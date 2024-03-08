import os

PORT = int(os.getenv("PORT", 80))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", 1000))
NICE = float(os.getenv("NICE", 0.5))
GPU = True if os.getenv("GPU", "true").lower == "true" else False
NODE_PATH = os.getenv("NODE_PATH", "node-data")

if not GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
from threading import Thread

from protopost import ProtoPost
from nd_to_json import nd_to_json

from aegis_mica import Node

#TODO: ignore link obs acquisition time in nice calc
#   might be difficult since the RL agent controls the link_env

node = Node.load(NODE_PATH)

save_steps = 0
steps = 0

#add "add_link" route

#set up protopost
def run_protopost():
    routes = {
        "": lambda x: nd_to_json(node.state),
    }

    ProtoPost(routes).start(PORT)

Thread(target=run_protopost, daemon=True).start()

while True:
    if len(node.links) == 0:
        print("Node has no links, sleeping...")
        time.sleep(1)
        continue

    t = time.time()
    mean_reward = node.update()
    dt = time.time() - t
    time.sleep(dt * NICE)

    steps += 1
    print(steps, mean_reward)

    #save
    save_steps += 1
    if save_steps >= SAVE_EVERY:
        print(f"Saving node to {NODE_PATH}...")
        node.save(NODE_PATH)
        save_steps = 0