#!/anaconda3/envs/pytorch/bin python3
import torch
import pypose as pp
import torch.nn as nn
import torch.nn.functional as F
import wandb

torch.set_default_dtype(torch.float32)

# visual-imperative-planning
from .tsdf_map import TSDF_Map
from .traj_opt import TrajOpt

class TrajCost:
    
    debug = False
    
    def __init__(
        self, 
        gpu_id=0, 
        sensorOffsetX=0.0,
        weight_difficult: float = 1.0, 
        weight_outside: float = 1.0,
        log_data: bool = False
    ) -> None:
        self.tsdf_map = TSDF_Map(gpu_id)
        self.opt = TrajOpt()
        self.is_map = False
        self.senserOffsetX = sensorOffsetX
        self.weight_difficult = torch.tensor(weight_difficult, dtype=torch.float32, device=torch.device("cuda:" + str(gpu_id)))
        self.weight_outside = torch.tensor(weight_outside, dtype=torch.float32, device=torch.device("cuda:" + str(gpu_id)))
        
        # logging
        self.log_data = log_data
        return

    def TransformPoints(self, odom, points):
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(batch_size, num_p, device=points.device, requires_grad=points.requires_grad)
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps
    
    def SetMap(self, root_path, map_name):
        self.tsdf_map.ReadTSDFMap(root_path, map_name)
        self.is_map = True
        return

    def CostofTraj(
        self,
        waypoints: torch.Tensor,
        odom: torch.Tensor,
        goal: torch.Tensor,
        log_step: int,
        ahead_dist: float,
        pair_difficult,
        pair_outside,
        w_obs=0.25,
        w_height=1.0,
        w_motion=1.5,
        w_goal=2.0,
        w_length=0,
        obstalce_thred=0.75,
        dataset: str = "train",
    ):
        batch_size, num_p, _ = waypoints.shape
        if self.is_map:
            waypoints[..., 0] = waypoints[..., 0] + self.senserOffsetX
            world_ps = self.TransformPoints(odom, waypoints)
            norm_inds, _ = self.tsdf_map.Pos2Ind(world_ps)
            
            # Obstacle Cost
            cost_grid = self.tsdf_map.cost_array.T.expand(batch_size, 1, -1, -1)
            oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
            oloss_M = oloss_M.to(torch.float32)
            oloss_M_weighted = torch.sum(oloss_M, axis=1)
            oloss_M_weighted[pair_difficult] = oloss_M_weighted[pair_difficult] * self.weight_difficult  # weighting
            oloss_M_weighted[pair_outside] = oloss_M_weighted[pair_outside] * self.weight_outside
            oloss = torch.mean(oloss_M_weighted)
            
            # Terrian Height loss
            height_grid = self.tsdf_map.ground_array.T.expand(batch_size, 1, -1, -1)
            hloss_M = F.grid_sample(height_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
            hloss_M = torch.abs(world_ps[:, :, 2]  - odom[:, None, 2] - hloss_M).to(torch.float32)  # world_ps - odom to have them on the ground to be comparable to the height map
            hloss_M = torch.sum(hloss_M, axis=1)
            hloss_M[pair_difficult] = hloss_M[pair_difficult] * self.weight_difficult  # weighting
            hloss_M[pair_outside] = hloss_M[pair_outside] * self.weight_outside
            hloss = torch.mean(hloss_M)
            
            if self.log_data:
                wandb.log({f"hloss_{dataset}_step": hloss}, step=log_step)
                wandb.log({f"oloss_{dataset}_step": oloss}, step=log_step)

        # Goal Cost - Control Cost
        gloss_M = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
        gloss_M_weighted = torch.clone(gloss_M)  # necessary otherwise verseion error
        gloss_M_weighted[pair_difficult] = gloss_M_weighted[pair_difficult] * self.weight_difficult  # weighting
        gloss_M_weighted[pair_outside] = gloss_M_weighted[pair_outside] * self.weight_outside
        gloss = torch.mean(gloss_M_weighted)
        
        # Moving Loss - punish staying 
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], step=1.0/(num_p-1)) 
        desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0:num_p-1, :], dim=2)
        wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0:num_p-1, :], dim=2)
        mloss = torch.abs(desired_ds - wp_ds)
        mloss = torch.sum(mloss, axis=1)
        mloss[pair_difficult] = mloss[pair_difficult] * self.weight_difficult  # weighting
        mloss[pair_outside] = mloss[pair_outside] * self.weight_outside
        mloss = torch.mean(mloss)

        # Path Length Loss
        lloss = torch.abs(torch.sum(wp_ds, axis=1) - torch.sum(desired_ds, axis=1)) 
        lloss[pair_difficult] = lloss[pair_difficult] * self.weight_difficult  # weighting
        lloss[pair_outside] = lloss[pair_outside] * self.weight_outside
        lloss = torch.mean(lloss)
        
        if self.log_data:
            wandb.log({f"gloss_{dataset}_step": gloss}, step=log_step)
            wandb.log({f"mloss_{dataset}_step": mloss}, step=log_step)
            wandb.log({f"lloss_{dataset}_step": lloss}, step=log_step)

        # Fear labels
        goal_dists = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
        floss_M = torch.clone(oloss_M)[:, 1:]
        floss_M[goal_dists > ahead_dist] = 0.0 
        fear_labels = torch.max(floss_M, 1, keepdim=True)[0]
        # fear_labels = nn.Sigmoid()(fear_labels-obstalce_thred)
        fear_labels = fear_labels > obstalce_thred
        
        # TODO: kinodynamics cost
        return w_obs*oloss + w_height*hloss + w_motion*mloss + w_goal*gloss + w_length*lloss, fear_labels.float()