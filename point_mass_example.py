import numpy as np
import torch

from torch_fabrics.fabrics import FabricHandler
from torch_fabrics.envs.point_mass_env import PointMassEnv

NUM_ROLLOUTS = 1000
SEED = 15
XML_PATH = None # TODO: place full XML path here

env = PointMassEnv(
    pos = torch.Tensor([0.1, 0.2]),
    vel = torch.Tensor([0.0, -0.1]),
    radius = 0.01,
    obs_pos = torch.Tensor([[0.1, 0.1], [-0.1, -0.05]]),
    obs_radii = 0.02 * torch.ones(2),
    goal = torch.Tensor([0.1, 0.0]),
    # goal = torch.Tensor([0.0, -0.6]),
    #goal = torch.Tensor([-0.15, -0.15]),
    xml_path = XML_PATH
)

fabric_handler = FabricHandler(env)

def policy(pos, vel):
    pos = torch.from_numpy(pos)
    vel = torch.from_numpy(vel)
    accel = fabric_handler.fabric_solve(pos, vel)
    return accel.detach().numpy()

state = env.reset()
action = policy(*state)

for _ in range(NUM_ROLLOUTS):
    state = env.reset()
    while True:
        accel = policy(*state)
        print("accel", accel)
        state, reward, done, _ = env.step(accel)
        env.render()
        if done:
            break
