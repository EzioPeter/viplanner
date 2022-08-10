#!/anaconda3/envs/pytorch/bin python3
import os
import torch
import pypose as pp
from tsdf_map import TSDF_Map
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Cuda is available.")
else:
    device = torch.device("cpu")
    print("Cuda is not available.")

def RandomPoints():
    num_p = 5
    batch_size = 1
    points_se3 = pp.randn_se3(batch_size, num_p, device=device) # random se3 points
    return points_se3


class CubicSplineTorch:
    # Reference: https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
    def __init__(self):
        return None

    def h_poly(self, t):
        alpha = torch.arange(4, device=t.device, dtype=t.dtype)
        tt = t[:, None, :]**alpha[None, :, None]
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
            ], dtype=t.dtype, device=t.device)
        return A @ tt

    def interp(self, x, y, xs):
        m = (y[:, 1:, :] - y[:, :-1, :]) / torch.unsqueeze(x[:, 1:] - x[:, :-1], 2)
        m = torch.cat([m[:, None, 0], (m[:, 1:] + m[:, :-1]) / 2, m[:, None, -1]], 1)
        idxs = torch.searchsorted(x[0, 1:], xs[0, :])
        dx = x[:, idxs + 1] - x[:, idxs]
        hh = self.h_poly((xs - x[:, idxs]) / dx)
        hh = torch.transpose(hh, 1, 2)
        out = hh[:, :, 0:1] * y[:, idxs, :]
        out = out + hh[:, :, 1:2] * m[:, idxs] * dx[:,:,None]
        out = out + hh[:, :, 2:3] * y[:, idxs + 1, :]
        out = out + hh[:, :, 3:4] * m[:, idxs + 1] * dx[:,:,None]
        return out

class TrajOpt:
    def __init__(self):
        self.tsdf_map = TSDF_Map()
        self.is_map = False
        self.cs_interp = CubicSplineTorch()

    def TrajGeneratorFromPFreeRot(self, preds, steps=0.1): 
        # Points is in se3
        batch_size, num_p, dims = preds.shape
        self.points_preds = torch.cat((torch.zeros(batch_size, 1, dims, device=preds.device, requires_grad=preds.requires_grad), preds), axis=1)
        num_p = num_p + 1
        xs = torch.arange(0, num_p-1+steps, steps, device=preds.device)
        xs = xs.repeat(batch_size, 1)
        x  = torch.arange(num_p, device=device, dtype=preds.dtype)
        x  = x.repeat(batch_size, 1)
        waypoints = self.cs_interp.interp(x, self.points_preds, xs)
        return waypoints  # R3

    def TransformPoints(self, odom, points):
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(batch_size, num_p, device=points.device, requires_grad=points.requires_grad)
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps


    def CostofTraj(self, waypoints, odom, goal, alpha=0.25, beta=1.0, gamma=1.5, delta=2.0): # cannot run with cuda
        batch_size, num_p, _ = waypoints.shape
        if self.is_map:
            world_ps = self.TransformPoints(odom, waypoints)
            norm_inds, _ = self.tsdf_map.Pos2Ind(world_ps)
            # Obstacle Cost
            cost_grid = self.tsdf_map.cost_array.T.expand(batch_size, 1, -1, -1)
            oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border').squeeze(1).squeeze(1)
            # oloss_M = F.pad(oloss_M, (padding-1, padding), "constant", 0.0)
            # oloss_M = F.avg_pool1d(oloss_M, kernel_size=kernel, stride=1, padding=0) * kernel
            # oloss_M = oloss_M + torch.sum(oloss_M, dim=1, keepdims=True) - torch.cumsum(oloss_M, dim=1) # reverse cumsum loss
            # oloss_M = torch.cumsum(oloss_M, axis=1)
            oloss = torch.mean(torch.sum(oloss_M, axis=1))
            
            # Terrian Height loss
            height_grid = self.tsdf_map.ground_array.T.expand(batch_size, 1, -1, -1)
            hloss_M = F.grid_sample(height_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border').squeeze(1).squeeze(1)
            hloss_M = torch.abs(waypoints[:, :, 2] - hloss_M)
            hloss = torch.mean(torch.sum(hloss_M, axis=1))

        # Goal Cost - Control Cost
        gloss = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
        gloss = torch.mean(gloss)
        
        # Moving Loss - punish staying 
        desired_wp = self.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], steps=1.0/(num_p-1)) 
        desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0:num_p-1, :], dim=2)
        wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0:num_p-1, :], dim=2)
        mloss = torch.abs(desired_ds - wp_ds)
        mloss = torch.sum(mloss, axis=1)
        mloss = torch.mean(mloss)

        # TODO: kinodynamics cost
        return alpha*oloss + beta*hloss + gamma*mloss + delta*gloss 
    
    def SetMap(self, root_path, map_name):
        self.tsdf_map.ReadTSDFMap(root_path, map_name)
        self.is_map = True
        return


if __name__ == '__main__':
    traj_opt = TrajOpt()
    env_name = "Matterport"
    root_path = os.path.join(*[os.getcwd(), "data", env_name])
    points_path = os.path.join(root_path, "points.txt")
    map_name = "tsdf1"
    odom = torch.tensor([0, 0, 0.5, 0.0, 0.0, 0.0, 1.0], device=device)
    points_se3 = RandomPoints()
    odom = odom.repeat(points_se3.shape[0],1)
    traj_opt.SetMap(root_path, map_name)
    waypoints = traj_opt.TrajGeneratorFromPFreeRot(points_se3, odom)
    # Compute cost of traj
    print("Current path cost: ", traj_opt.CostofTraj(waypoints).item())