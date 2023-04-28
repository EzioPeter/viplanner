#!/anaconda3/envs/pytorch/bin python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

torch.set_default_dtype(torch.float32)

# visual-imperative-planning
from .traj_opt import TrajOpt
from viplanner.cost_maps import CostMapPCD
try:
    import pypose as pp  # only used for training
    import wandb  # only used for training
except ModuleNotFoundError or ImportError:  # eval in issac sim  # TODO: check if all can be installed in Isaac Sim
    print("[Warning] pypose or wandb not found, only use for evaluation")


class TrajCost:
    
    debug = False
    
    def __init__(
        self, 
        gpu_id: Optional[int] = 0, 
        log_data: bool = False,
        w_obs: float = 0.25,
        w_height: float = 1.0,
        w_motion: float = 1.5,
        w_goal: float = 2.0,
        obstalce_thred: float = 0.75,
    ) -> None:
        # init map and optimizer
        self.gpu_id = gpu_id
        self.cost_map: CostMapPCD = None
        self.opt = TrajOpt()
        self.is_map = False
        
        # loss weights
        self.w_obs = w_obs
        self.w_height = w_height
        self.w_motion = w_motion
        self.w_goal = w_goal
        
        # fear label threshold value
        self.obstalce_thred = obstalce_thred
        
        # logging
        self.log_data = log_data
        return

    @staticmethod
    def TransformPoints(odom, points):
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(batch_size, num_p, device=points.device, requires_grad=points.requires_grad)
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps
    
    def SetMap(self, root_path, map_name):
        self.cost_map = CostMapPCD.ReadTSDFMap(root_path, map_name, self.gpu_id)
        self.is_map = True
        return

    def CostofTraj(
        self,
        waypoints: torch.Tensor,
        odom: torch.Tensor,
        goal: torch.Tensor,
        log_step: int,
        ahead_dist: float,
        dataset: str = "train",
    ):
        batch_size, num_p, _ = waypoints.shape
        
        assert self.is_map, "Map has to be set for cost calculation"
        world_ps = self.TransformPoints(odom, waypoints)
        norm_inds, _ = self.cost_map.Pos2Ind(world_ps)
        
        # Obstacle Cost
        cost_grid = self.cost_map.cost_array.T.expand(batch_size, 1, -1, -1)
        oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
        oloss_M = oloss_M.to(torch.float32)
        oloss_M_weighted = torch.sum(oloss_M, axis=1)
        oloss = torch.mean(oloss_M_weighted)
        
        if self.debug:
            import numpy as np
            # indexes in the cost map
            start_xy = torch.tensor([self.cost_map.start_x, self.cost_map.start_y], dtype=torch.float64, device=world_ps.device).expand(1, 1, -1)
            H = (world_ps.tensor()[:, :, 0:2] - start_xy) / self.cost_map.voxel_size
            cost_values = self.cost_map.cost_array[H[0, :, 0].cpu().numpy().astype(np.int64), H[0, :, 1].cpu().numpy().astype(np.int64)]

            import matplotlib.pyplot as plt
            fix, (ax1, ax2, ax3) = plt.subplots(1, 3)
            sc1 = ax1.scatter(world_ps.data[0][:, 0].cpu().numpy(), world_ps.data[0][:, 1].cpu().numpy(), c=oloss_M[0].cpu().numpy(), cmap='rainbow')
            sc2 = ax2.scatter(H[0, :, 0].cpu().numpy(), H[0, :, 1].cpu().numpy(), c=cost_values.cpu().numpy(), cmap='rainbow')
            ax3.imshow(self.cost_map.cost_array.cpu().numpy())
            
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world_ps.data[0][:, :3].cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(sc1.to_rgba(oloss_M[0].cpu().numpy())[:, :3])
            # pcd.colors = o3d.utility.Vector3dVector(sc2.to_rgba(cost_values[0].cpu().numpy())[:, :3])
            o3d.visualization.draw_geometries([self.cost_map.pcd_tsdf, pcd])
            
        # Terrian Height loss
        height_grid = self.cost_map.ground_array.T.expand(batch_size, 1, -1, -1)
        hloss_M = F.grid_sample(height_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
        hloss_M = torch.abs(world_ps[:, :, 2]  - odom[:, None, 2] - hloss_M).to(torch.float32)  # world_ps - odom to have them on the ground to be comparable to the height map
        hloss_M = torch.sum(hloss_M, axis=1)
        hloss = torch.mean(hloss_M)
        
        if self.log_data:
            wandb.log({f"hloss_{dataset}_step": hloss}, step=log_step)
            wandb.log({f"oloss_{dataset}_step": oloss}, step=log_step)

        # Goal Cost - Control Cost
        gloss_M = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
        gloss = torch.mean(gloss_M)
        
        # Moving Loss - punish staying 
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], step=1.0/(num_p-1)) 
        desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0:num_p-1, :], dim=2)
        wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0:num_p-1, :], dim=2)
        mloss = torch.abs(desired_ds - wp_ds)
        mloss = torch.sum(mloss, axis=1)
        mloss = torch.mean(mloss)
        
        if self.log_data:
            wandb.log({f"gloss_{dataset}_step": gloss}, step=log_step)
            wandb.log({f"mloss_{dataset}_step": mloss}, step=log_step)

        # Fear labels
        goal_dists = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
        floss_M = torch.clone(oloss_M)[:, 1:]
        floss_M[goal_dists > ahead_dist] = 0.0 
        fear_labels = torch.max(floss_M, 1, keepdim=True)[0]
        # fear_labels = nn.Sigmoid()(fear_labels-obstalce_thred)
        fear_labels = fear_labels > self.obstalce_thred
        
        # TODO: kinodynamics cost
        return self.w_obs*oloss + self.w_height*hloss + self.w_motion*mloss + self.w_goal*gloss, fear_labels.float()
    
    def cost_of_recorded_path(
        self,
        waypoints: torch.Tensor,
    ) -> None:
        """Cost of recorded path - for evaluation only

        Args:
            waypoints (torch.Tensor): Path coordinates in world frame
            goal (torch.Tensor): Gaol coordinates in world frame
        """        
        assert self.is_map, "Map has to be loaded for evaluation"
        norm_inds, _ = self.cost_map.Pos2Ind(waypoints)
        
        # Obstacle Cost
        cost_grid = self.cost_map.cost_array.T
        oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
        oloss_M = oloss_M.to(torch.float32)
        oloss_M_weighted = torch.sum(oloss_M, axis=1)
        oloss = torch.mean(oloss_M_weighted)
        
        return oloss
    
# EoF
