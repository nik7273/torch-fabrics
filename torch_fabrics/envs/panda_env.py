import os
import numpy as np
import torch

from torch import cos
from torch import sin
from functorch import jacfwd, hessian, grad
from typing import List
from copy import copy, deepcopy
from dm_control import mjcf

from torch_fabrics.envs.robot_env import RobotEnv
from gymnasium.spaces import Box

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

class PandaEnv(RobotEnv):
    def __init__(
        self, 
        pos: torch.Tensor,
        goal: torch.Tensor,
        init_config: torch.Tensor,
        init_vel: torch.Tensor,
        obs_pos: torch.Tensor,
        obs_radii: torch.Tensor,
        xml_path: str = "models/panda/franka_panda.xml"
    ):
        assert len(obs_pos) == len(obs_radii)
        self.init_pos = pos
        self.init_config = init_config
        self.init_vel = init_vel
        self.obs_pos = obs_pos
        self.obs_radii = obs_radii
        self.goal_pos = goal
        
        self.point = torch.Tensor([0,0,0,1])
        # modified DH parameters
        self.dh_params = {
            'r': torch.Tensor([0., 0., 0., 0.0825, -0.0825, 0., 0.088, 0]),
            'd': torch.Tensor([0.333, 0., 0.316, 0., 0.384, 0., 0., 0.107]),
            'alpha': torch.Tensor(
                [0., -torch.pi/2, torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2, torch.pi/2, 0.]
            )
        }

        self.t = 0
        obs_shape = 18;
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        super().__init__(xml_path, simrate=60, observation_space=observation_space)

    def step(self, action: np.ndarray):
        self.t += 1
        self.step_sim(action)
        obs = (self.data.qpos, self.data.qvel)
        reward = self.get_reward()
        qpos = torch.Tensor(self.data.qpos)
        ee_pos = self._apply_fk(qpos)[0][-1] + torch.Tensor([0, 0, 0.0584])
        # simple point navigation
        # dynamically specifying done based on fabric could be interesting...
        done = torch.linalg.norm(self.goal_pos - ee_pos) < 0.02
        print(f"t: {self.t}")
        # done = done or self.t > 200
        return obs, reward, done, {}

    def reset(self):
        self.t = 0
        self.set_state(
            self.init_config.detach().numpy(),
            self.init_vel.detach().numpy()
        )
        qpos = np.copy(self.data.qpos)
        qvel = np.copy(self.data.qvel)
        return qpos, qvel

    def get_reward(self):
        return 0

    ############# FABRIC STUFF ####################

    def _fk_matrix(self, theta: torch.Tensor, i: int):
        # modified DH params
        dh = self.dh_params
        r = dh['r'][i]
        d = dh['d'][i]
        alpha = dh['alpha'][i]
        T = torch.Tensor([
            [0.,0.,0.,0.],
            [0.,0.,0.,0.],
            [0.,0.,0.,0.],
            [0.,0.,0.,1.],
        ])
        # this way keeps differentiability
        T[0][0] = cos(theta)
        T[0][1] = -sin(theta)
        T[0][2] = 0
        T[0][3] = r
        T[1][0] = sin(theta)*cos(alpha)
        T[1][1] = cos(theta)*cos(alpha)
        T[1][2] = -sin(alpha)
        T[1][3] = -d*sin(alpha)
        T[2][0] = sin(theta)*sin(alpha)
        T[2][1] = cos(theta)*sin(alpha)
        T[2][2] = cos(alpha)
        T[2][3] = d*cos(alpha)

        return T

    def _apply_fk(self, qpos: torch.Tensor):
        # forward kinematics to get transformations for every joint
        # all_Ts stores transformation from joint i to world frame
        all_Ts = []
        running_T = torch.eye(4, requires_grad=True)
        relevant_qpos = qpos[:7]
        for i, q in enumerate(relevant_qpos):
            T = self._fk_matrix(q, i)
            running_T = running_T @ T
            all_Ts.append(running_T)

        # flange is at pi/4? adjusting for lack of info in XML
        T = self._fk_matrix(torch.Tensor([torch.pi/4]), 7)
        running_T = running_T @ T
        all_Ts.append(running_T)
            
        # after getting all transformations, extract position coordinates
        # these coordinates are our link poses
        link_poses = [
            T_mat @ self.point for T_mat in all_Ts
        ]
        link_poses = [
            pose[:3] for pose in link_poses
        ]
        return link_poses, all_Ts

    def attractor_task_map(self, qpos: torch.Tensor):
        # easy, non-differentiable way:
        # diff btwn panda_hand and goal position
        # return self.sim.data.body_xpos[9] - self.goal_pos

        # differentiable way is to use the FK map
        link_poses, all_Ts = self._apply_fk(qpos)
        # add gripper pose because not in XML
        flange_pose = link_poses[-1]
        gripper_pose = flange_pose + torch.Tensor([0, 0, 0.0584]).double()
        return gripper_pose - self.goal_pos

    def obstacle_avoidance_task_map(self, qpos: torch.Tensor):
        out = []
        link_poses, _ = self._apply_fk(qpos)
        n_pose = len(link_poses)
        n_obs = len(self.obs_pos)

        diff = (torch.stack(link_poses)[:, None, :] - self.obs_pos[None, :, :]).view(-1, 3)
        radii = (0.5 * self.obs_radii).view(1, -1)
        norms = torch.linalg.norm(diff, axis=1)
        if n_obs > 1:
            radii = radii.repeat(n_pose, n_obs).view(-1)
            norms = norms.repeat_interleave(n_obs)
            out = (norms/radii)[::n_obs] - 1
        else:
            out = norms / radii - 1

        return out.double().view(-1)

        # out = []
        # link_poses, _ = self._apply_fk(qpos)
        # n_pose = len(link_poses)
        # n_obs = len(self.obs_pos)

        # scalar = 0.5
        # for i, obs in enumerate(self.obs_pos):
        #     radius = scalar * self.obs_radii[i]
        #     for link in link_poses:
        #         x = (torch.linalg.norm(link - obs) / radius) - 1.0
        #         out.append(x)
        
        # return torch.stack(out).double()

    def floor_repulsion_task_map(self, qpos: torch.Tensor):
        link_poses, all_Ts = self._apply_fk(qpos)
        return torch.stack(link_poses)[:,2].double()
        # add gripper pose because not in XML
        # flange_pose = link_poses[-1]
        # gripper_pose = flange_pose + torch.Tensor([0, 0, 0.0584]).double()        
        # return gripper_pose - self.goal_pos

    def upper_joint_limit_task_map(self, qpos: torch.Tensor):
        upper_limit = self.sim.model.jnt_range[:7,0]
        return torch.Tensor(upper_limit) - qpos[:7]

    def lower_joint_limit_task_map(self, qpos: torch.Tensor):
        lower_limit = self.sim.model.jnt_range[:7,1]
        return torch.Tensor(lower_limit) - qpos[:7]

    def attractor_fabric(self, pos: torch.Tensor, vel: torch.Tensor):
        """
        k = potential scaling term, increase to increase magnitude of acceleration
        upper_m = upper isotropic mass, 
        lower_m = lower isotropic mass,
        radial_basis_width = 
        transition_rate = 
        damping = velocity damping term
        """
        k = 15000000.0
        upper_m = 200.0
        lower_m = 10.0
        radial_basis_width = 5.0
        transition_rate = 10.0
        damping = 15.0

        potential = lambda x: k*(torch.linalg.norm(x) + (1/transition_rate)*torch.log(1+torch.exp(-2*transition_rate*torch.linalg.norm(x))))
        pos_ = pos.clone()
        potential_grad = grad(potential)(pos_)

        accel = -potential_grad - damping*vel
        metric = (upper_m - lower_m)*torch.exp(-(radial_basis_width * torch.norm(pos))**2)*torch.eye(3) + lower_m*torch.eye(3)

        print(f"Accel attractor: {accel}")
        print(f"Metric attractor: {metric}")
        return metric, accel

    def obstacle_avoidance_fabric(self, pos: torch.Tensor, vel: torch.Tensor):
        """
        Parameters:
        k_b = metric scaling term
        a_b = potential scaling term
        """
        k_b = 15000
        a_b = 1500000# potential scaling term

        s = (vel < 0).double()
        potential = lambda x: a_b / (2*(x**8))
        pos_ = pos.clone()
        potential_grad = jacfwd(potential)(pos_)
        
        accel = -s * (vel**2) @ potential_grad.double()
        metric = torch.diag(s) * k_b / (pos**2)

        print(f"Accel repeller: {accel}")
        print(f"Metric repeller: {metric}")

        return metric, accel
    
    def floor_repulsion_fabric(self, pos: torch.Tensor, vel: torch.Tensor):
        """
        Parameters:
        k_b = metric scaling term
        a_b = potential scaling term
        """
        k_b = 150
        a_b = 150# potential scaling term

        s = (vel < 0).double()
        potential = lambda x: a_b / (2*(x**8))
        pos_ = pos.clone()
        potential_grad = jacfwd(potential)(pos_)
        
        accel = -s * (vel**2) @ potential_grad.double()
        metric = torch.diag(s) * k_b / (pos**2)

        print(f"Accel repeller: {accel}")
        print(f"Metric repeller: {metric}")

        return metric, accel

    def joint_limit_fabric(self, pos: torch.Tensor, vel: torch.Tensor):
        """
        Parameters:
        l = 
        a1 = 
        a2 = 
        a3 = 
        a4 = 
        """
        pos = pos[:7]
        vel = vel[:7]
        l = 0.25
        a1, a2, a3, a4 = 0.3, 0.5, 1.0, 2.0
        s = (vel < 0).double()
        metric = torch.diag(s * (l / pos))
        potential = lambda q: (a1 / q**2) + a2 * torch.log(1. + torch.exp(-a3 * (q - a4)))
        pos_ = pos.clone()
        potential_grad = jacfwd(potential)(pos_)
        accel = -s * torch.linalg.norm(vel)**2 @ potential_grad

        return metric, accel

    def _set_fabrics(self):
        self.task_names = [
            "attractor",
            "obstacle_avoidance",
#            "floor_repulsion",
#            "upper_jointlim",
#            "lower_jointlim"
        ]
        self.task_maps = {
            "attractor" : self.attractor_task_map,
            "obstacle_avoidance" : self.obstacle_avoidance_task_map,
            "floor_repulsion": self.floor_repulsion_task_map,
            "upper_jointlim" : self.upper_joint_limit_task_map,
            "lower_jointlim" : self.lower_joint_limit_task_map
        }
        self.fabrics = {
            "attractor" : self.attractor_fabric,
            "obstacle_avoidance" : self.obstacle_avoidance_fabric,
            "floor_repulsion" : self.floor_repulsion_fabric,
            # # double weighting? whatever
            "upper_jointlim" : self.joint_limit_fabric,
            "lower_jointlim" : self.joint_limit_fabric
        }
