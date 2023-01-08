import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from torch_fabrics.fabrics import FabricHandler
from torch_fabrics.envs.panda_env import PandaEnv

NUM_ROLLOUTS = 10
SEED = 15
XML_PATH = None # TODO: place full XML path here

env = PandaEnv(
    pos=torch.Tensor([0.1, 0.5, 0.3]),
    goal=torch.Tensor([-0.3, 0.4, 0.3]),
    init_config=torch.Tensor([
        0.5,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        0.02,
        0.02
    ]), # based on jnt_range
    init_vel = torch.Tensor([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ]),
    obs_pos=torch.Tensor([
        [0.0, -0.4, 0.3],
        [-0.5, -0.4, 0.3]
    ]),
    obs_radii=torch.Tensor([0.05, 0.05]),
    xml_path=XML_PATH
)

fabric_handler = FabricHandler(env)

def policy(pos: np.ndarray, vel: np.ndarray):
    pos = torch.from_numpy(pos)
    vel = torch.from_numpy(vel)
    accel = fabric_handler.fabric_solve(pos, vel)
    return accel.detach().numpy()

state = env.reset()
action = policy(*state)

writer = SummaryWriter()

i = 0
for _ in range(NUM_ROLLOUTS):
    state = env.reset()
    while True:
        i += 1
        accel = policy(*state)
        # accel = np.zeros_like(env.data.qvel)
        state, reward, done, _ = env.step(accel)
        
        env.render()
        if done:
            break
