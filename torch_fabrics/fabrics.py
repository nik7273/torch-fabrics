import torch
from functorch import jacfwd, jvp

from torch_fabrics.envs.robot_env import RobotEnv

class FabricHandler:
    def __init__(self, env: RobotEnv):
        self.env = env
    
    def fabric_solve(self, pos: torch.Tensor, vel: torch.Tensor):        
        # pos is joint pos in configuration space
        # vel is joint vel in configuration space
        # enforce: pos and vel same size
        pos = pos.detach().clone()
        pos.requires_grad_()
        vel = vel.detach().clone()
        vel.requires_grad_()
        all_metric, all_accel, all_c, all_J = [], [], [], []
        # all_v_, all_a_ = [], [], [], [], []
        q_ = pos.detach().clone()
        q__ = pos.detach().clone()
        for task_name in self.env.task_names: # function handles
            task_map = self.env.task_maps[task_name]
            # leaf pos
            x = task_map(pos)
            # this is the naive version!
            J = jacfwd(task_map)(pos)
            # leaf vel
            v = jvp(task_map, (pos,), (vel,))[1]
            # v = jvp(task_map, pos, v=vel, create_graph=True, strict=True)[1]            
            # curvature terms
            inner_fn = lambda q: jvp(task_map, (q,), (vel,))[1]
            # strict is false because it's possible for the result to be independent,
            # in which case we just do what we can
            c = jvp(inner_fn, (pos,), (vel,))[1]
            # fabric
            metric, accel = self.fabric_eval(x, v, task_name)

            metric = metric.to(dtype=torch.float64)
            accel = accel.to(dtype=torch.float64)

            # backward pass
            # v_ = task_map(q_)
            # a_ = task_map(q__)

            all_J.append(J)
            all_metric.append(metric)
            all_accel.append(accel)
            all_c.append(c)
            # all_v_.append(v_)
            # all_a_.append(a_)

        accel_c = [acc - c for (acc, c) in zip(all_accel, all_c)]

        # einsum can be used as a replacement
        M_r = sum([torch.transpose(J, 0, 1) @ M @ J for (J,M) in zip(all_J, all_metric)])
        f_r = sum([torch.transpose(J, 0, 1) @ M @ acc_c for (J,M,acc_c) in zip(all_J, all_metric, accel_c)])

        # energize fabric?
        # vv = vel
        # v_h = vv / torch.linalg.norm(vv)
        # I = torch.eye(vel.shape[0])
        # second = I - v_h @ v_h.unsqueeze(1)
        # P_e = second # assume M_e = I, f_e = 0 using euclidean energy

        accel = torch.linalg.pinv(M_r) @ f_r

        # accel_energized = -P_e @ accel    

        # accel = torch.linalg.pinv(M_energized) @ f_r
        # return accel_energized
        return accel

    def fabric_eval(self, pos: torch.Tensor, vel: torch.Tensor, task_name: str):
        fabric_handle = self.env.fabrics[task_name]
        metric, accel = fabric_handle(pos, vel)
        return metric, accel

