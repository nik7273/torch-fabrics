import os
import numpy as np
import torch
from functorch import jacfwd, grad, hessian
from typing import List
from pathlib import Path
from dm_control import mjcf
from gymnasium.spaces import Box

from torch_fabrics.envs.robot_env import RobotEnv

class PointMass:
    """A point mass"""
    def __init__(self, radius=0.01, xml_path=None):
        
        self.mjcf_model = mjcf.RootElement(model="point_mass")

        # self.mjcf_model.option.timestep = 0.02
        
        # Assets
        self.self_mat = self.mjcf_model.asset.add(
            'material',
            name='self',
            rgba=[0.7,0.5,0.3,0.1]
        )

        # Defaults
        self.mjcf_model.default.joint.type = 'hinge'
        self.mjcf_model.default.joint.damping = 1
        self.mjcf_model.default.joint.axis = [0,0,1]
        self.mjcf_model.default.joint.limited = True
        self.mjcf_model.default.joint.range = [-.79, .79]
        self.mjcf_model.default.motor.gear = [0.1]
        self.mjcf_model.default.motor.ctrlrange = [-1, 1]
        self.mjcf_model.default.motor.ctrllimited = True

        # Mass
        self.pointmass = self.mjcf_model.worldbody.add(
            'body',
            pos=[0,0,radius]
        )
        self.root_x = self.pointmass.add(
            'joint', 
            name='root_x',
            type='slide',
            pos=[0,0,0],
            axis=[1,0,0]
        )
        self.root_y = self.pointmass.add(
            'joint', 
            name='root_y',
            type='slide',
            pos=[0,0,0],
            axis=[0,1,0]
        )
        self.geom = self.pointmass.add(
            'geom',
            type='sphere',
            size=[radius],
            material=self.self_mat,
            mass=0.3
        )
        
        # Actuators
        self.mjcf_model.actuator.add('motor', name='t1', joint=self.root_x)
        self.mjcf_model.actuator.add('motor', name='t2', joint=self.root_y)
        # self.mjcf_model.actuator.add('position', name='t1_pos', kp=.5, joint=self.root_x)
        # self.mjcf_model.actuator.add('velocity', name='t1_vel', kv=4, joint=self.root_x)
        # self.mjcf_model.actuator.add('position', name='t2_pos', kp=.5, joint=self.root_y)
        # self.mjcf_model.actuator.add('velocity', name='t2_vel', kv=4, joint=self.root_y)

class SphereObstacle:
    def __init__(self, pos=[0,0,0.01], radius=0.015, name=None):
        if len(pos) == 2:
            pos = [pos[0], pos[1], radius]
        pos[2] = radius

        self.mjcf_model = mjcf.RootElement(model=name)
        self.obstacle_mat = self.mjcf_model.asset.add(
            'material',
            name='obstacle',
            rgba=[0.6, 0.3, 0.3, 1]
        )
        self.mjcf_model.worldbody.add(
            'geom',
            name='obstacle',
            pos=pos,
            material=self.obstacle_mat,
            type='sphere',
            size=[radius]
        )

class World:
    def __init__(self, init_pos, goal_pos, camera_pos=[0,0,0.75]):
        self.mjcf_model = mjcf.RootElement(model="world")
        self.light = self.mjcf_model.worldbody.add("light", pos=[0,0,1])

        init_pos = [init_pos[0], init_pos[1], 0.1]
        goal_pos = [goal_pos[0], goal_pos[1], 0.1]

        # visual stuff
        self.mjcf_model.visual.headlight.ambient = [0.4, 0.4, 0.4]
        self.mjcf_model.visual.headlight.diffuse = [0.8, 0.8, 0.8]
        self.mjcf_model.visual.headlight.specular = [0.1, 0.1, 0.1]
        self.mjcf_model.visual.map.znear = 0.01
        self.mjcf_model.visual.quality.shadowsize = 2048

        self.skybox_tex = self.mjcf_model.asset.add(
            'texture',
            name='skybox',
            type='skybox',
            builtin='gradient',
            rgb1=[0.4, 0.6, 0.8],
            rgb2=[0,0,0],
            width=800,
            height=800,
            mark="random",
            markrgb=[1,1,1]
        )
        self.grid_tex = self.mjcf_model.asset.add(
            'texture',
            name='grid',
            type='2d',
            builtin='checker',
            rgb1=[0.1,0.2,0.3],
            rgb2=[0.2,0.3,0.4],
            width=300,
            height=300,
            mark='edge',
            markrgb=[0.2,0.3,0.4]
        )
        self.grid_mat = self.mjcf_model.asset.add(
            'material',
            name='grid',
            texrepeat=[1,1],
            texuniform=True,
            reflectance=0.2
        )
        self.camera = self.mjcf_model.worldbody.add(
            'camera', 
            name='fixed',
            pos=camera_pos,
            quat=[1,0,0,0]
        )
        self.ground = self.mjcf_model.worldbody.add(
            'geom',
            name='ground',
            type='plane',
            pos=[0,0,0],
            size=[0.8,0.8,0.1],
            material=self.grid_mat
        )                
        self.dec_mat = self.mjcf_model.asset.add(
            'material',
            name='decoration',
            rgba=[0.3,0.5,0.7,1]
        )        
        self.wall_x = self.mjcf_model.worldbody.add(
            'geom',
            name='wall_x',
            type='plane',
            pos=[-0.8, 0, 0.02],
            zaxis=[1,0,0],
            size=[0.02, 0.8, 0.02],
            material=self.dec_mat
        )        
        self.wall_y = self.mjcf_model.worldbody.add(
            'geom',
            name='wall_y',
            type='plane',
            pos=[0, -0.8, 0.02],
            zaxis=[0,1,0],
            size=[0.8, 0.02, 0.02],
            material=self.dec_mat
        )        
        self.wall_neg_x = self.mjcf_model.worldbody.add(
            'geom',
            name='wall_neg_x',
            type='plane',
            pos=[0.8, 0, 0.02],
            zaxis=[-1,0,0],
            size=[0.02, 0.8, 0.02],
            material=self.dec_mat
        )
        self.wall_neg_y = self.mjcf_model.worldbody.add(
            'geom',
            name='wall_neg_y',
            type='plane',
            pos=[0, 0.8, 0.02],
            zaxis=[0,-1,0],
            size=[0.8, 0.02, 0.02],
            material=self.dec_mat
        )
        self.init_mat = self.mjcf_model.asset.add(
            'material',
            name='init_marker',
            rgba=[0, 0, 1, 0.5]
        )
        self.init_marker = self.mjcf_model.worldbody.add(
            'geom',
            name='init_marker',
            type='sphere',
            pos=init_pos,
            size=[0.01],
            material=self.init_mat
        )
        self.goal_mat = self.mjcf_model.asset.add(
            'material',
            name='goal_marker',
            rgba=[0, 1, 0, 0.5]
        )
        self.goal_marker = self.mjcf_model.worldbody.add(
            'geom',
            name='goal_marker',
            type='sphere',
            pos=goal_pos,
            size=[0.01],
            material=self.goal_mat
        )


class PointMassEnv(RobotEnv):
    def __init__(
        self,
        pos: torch.Tensor = None,
        vel: torch.Tensor = None,
        radius: float = None,
        obs_pos: torch.Tensor = None,
        obs_radii: List[float] = None,
        goal: torch.Tensor = None,
        xml_path: str = "models/curr_pointmass.xml"
    ):
        self.init_pos = pos
        self.init_pos_3d = np.array([pos[0], pos[1], 0])
        self.init_vel = vel
        self.radius = radius
        self.obs_pos = obs_pos
        self.obs_radii = obs_radii
        self.goal = goal
        self.goal_pos_3d = np.array([goal[0], goal[1], 0])

        dir_path = os.path.dirname(os.path.realpath(__file__))
        full_path = xml_path # lol hard code, whatever
        self._create_xml(radius, full_path)
        
        self.t = 0
        obs_shape = 4;
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        super().__init__(xml_path, simrate=60, observation_space=observation_space)

    def _create_xml(self, pm_rad, out_path: str = None):
        world = World(self.init_pos, self.goal)
        point_mass = PointMass(radius=pm_rad)
        world.mjcf_model.attach(point_mass.mjcf_model)
        for i, (op, r) in enumerate(zip(self.obs_pos, self.obs_radii)):
            sphere = SphereObstacle(pos=op, radius=r, name=f"sphere_{i}")
            world.mjcf_model.attach(sphere.mjcf_model)

        with open(out_path, 'w') as outfile:
            xml_str = world.mjcf_model.to_xml_string()
            outfile.write(xml_str)

    def step(self, action: np.ndarray):
        self.t += 1
        self.step_sim(action)        
        full_body_xpos = self.data.xpos
        pm_pos = full_body_xpos[2, :2]
        obs = (self.data.qpos, self.data.qvel)
        reward = self.get_reward()
        print(f"t: {self.t}")
        done = np.linalg.norm(self.goal.numpy() - pm_pos) < 0.01
        done = done or self.t > 300
        return obs, reward, done, {}

    def reset(self):
        self.t = 0
        self.set_state(self.init_pos, self.init_vel)        
        full_body_xpos = self.data.xpos
        pm_pos = full_body_xpos[2, :2]
        return (pm_pos, self.data.qvel)

    def get_reward(self):
        full_body_xpos = self.data.xpos
        pm_pos = full_body_xpos[2, :2]
        return np.abs(pm_pos - self.goal.numpy())

    # fabrics-specific stuff
    def repeller_task_map(self, pos: torch.Tensor):
        out = []
        for i, obs in enumerate(self.obs_pos):
            radius = 0.5 * self.obs_radii[i]
            x = (torch.linalg.norm(pos - obs) / radius) - 1.0
            out.append(x)
        
        return torch.stack(out)

    def attractor_task_map(self, pos: torch.Tensor):
        return pos - self.goal

    def attractor_fabric(self, pos: torch.Tensor, vel: torch.Tensor):
        k = 1000.0
        upper_m = 2.0
        lower_m = 0.2
        radial_basis_width = 1.3
        transition_rate = 10.0
        damping = 1.5

        potential = lambda x: k*(torch.linalg.norm(x) + (1/transition_rate)*torch.log(1+torch.exp(-2*transition_rate*torch.linalg.norm(x))))
        pos_ = pos.clone()
        potential_grad = grad(potential)(pos_)

        accel = -potential_grad - damping*vel
        metric = (upper_m - lower_m)*torch.exp(-(radial_basis_width * torch.norm(pos))**2)*torch.eye(2) + lower_m*torch.eye(2)

        print(f"Accel: {accel}")
        print(f"Metric: {metric}")

        return metric, accel

    def repeller_fabric(self, pos: torch.Tensor, vel: torch.Tensor):
        k_b = 0.1 # 75.0
        a_b = 0.1 #50.0 # potential scaling term

        s = (vel < 0).double()
        potential = lambda x: a_b / (2*(x**8))
        pos_ = pos.clone()
        potential_grad = jacfwd(potential)(pos_)

        accel = -s * (vel**2) @ potential_grad
        metric = torch.diag(s) * k_b / (pos**2)

        return metric, accel

    def _set_fabrics(self):
        self.task_names = [
            "attractor", 
            "repeller"
        ]
        self.task_maps = {
            "attractor" : self.attractor_task_map,
            "repeller" : self.repeller_task_map
        }
        self.fabrics = {
            "attractor" : self.attractor_fabric,
            "repeller" : self.repeller_fabric
        }
