import numpy as np
import mujoco
from abc import abstractmethod
from gym import utils
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Space

class RobotEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 8,
    }

    def __init__(
        self,
        xml_path: str = None,
        simrate: int = 60,
        observation_space: Space = None,
    ):
        MujocoEnv.__init__(self, xml_path, simrate, observation_space=observation_space, render_mode='human')
        utils.EzPickle.__init__(self)

        self.model.opt.timestep = 0.001

        self.task_names = []
        self.task_maps = {}
        self.fabrics = {}

        self.min_jnt_torque_lims = self.model.actuator_forcerange[:,0]
        self.max_jnt_torque_lims = self.model.actuator_forcerange[:,1]

        # TODO: remove hardcoded limits and gains
        # self.pos_lower_limit = self.sim.model.jnt_range[0, :]
        # self.pos_upper_limit = self.sim.model.jnt_range[1, :]
        # self.vel_lower_limit = -150
        # self.vel_upper_limit = 150
        # self.kd = 0.25
        # self.kp = 0.5

        # self.dt_ = 0.0001
        self._set_fabrics()

        self.timestep = 0 

        # self.xml_path = xml_path
        # self.simrate = simrate

    def step_sim(self, action: np.ndarray):
        # action is an acceleration!
        qpos = self.data.qpos
        qvel = self.data.qvel

        # if using torques
        H = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, H, self.data.qM)

        H = H.reshape((-1, action.shape[0]))

        torque_ctrl = np.dot(H, action)

        np.clip(
            torque_ctrl, 
            self.min_jnt_torque_lims, 
            self.max_jnt_torque_lims,
            out=torque_ctrl
        )

        print(f"Torque ctrl: {torque_ctrl}")

        self.do_simulation(torque_ctrl, 3)

    @abstractmethod
    def _set_fabrics(self):
        pass
