#!/usr/bin/env python3
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch
@author     Fan Yang
@email      fanyang1@ethz.ch

@brief      prepare dataset for VIPlanner training and inference
"""

# python
import os
import math
import torch
import numpy as np
import pypose as pp
import PIL
import copy
import time
from PIL import Image
from random import sample
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm
import cv2
import scipy.spatial.transform as tf
import open3d as o3d
import torch.nn.functional as F
import torchvision.transforms as transforms
import warp as wp
wp.init()

# implerative-planner-learning
from utils.warp_utils import _raycast
from config import DataCfg
from traj_cost_opt import TSDF_Map

# set default dtype to float32
torch.set_default_dtype(torch.float32)


class PlannerData(Dataset):
    def __init__(
        self,
        cfg: DataCfg,
        transform,
        semantics: bool = False,
    ) -> None:
        """_summary_

        Args:
            cfg (DataCfg): Dataset COnfiguration
            transform (_type_): Compose torchvision transforms (resize and to tensor)
            semantics (bool, optional): If semantics are used in the network input. Defaults to False.
        """
        
        self._cfg = cfg
        self.transform = transform
        self.semantics = semantics

        # vertical flip transform
        self.flip_transform = transforms.RandomVerticalFlip(p=1.0)

        # init buffers
        self.depth_filename: List[str] = []
        self.sem_filename: List[str] = []
        self.odom: torch.Tensor = None
        self.goal: torch.Tensor = None
        self.pair_difficult: np.ndarray = None
        self.pair_outside: np.ndarray = None  
        self.pair_augment: np.ndarray = None
        self.fov_angle: float = 0.0
        return     

    def update_buffers(
        self,
        depth_filename: List[str],
        sem_filename: List[str],
        odom: torch.Tensor,
        goal: torch.Tensor,
        pair_difficult: np.ndarray,
        pair_outside: np.ndarray,
        pair_augment: np.ndarray,
    ) -> None:
        
        self.depth_filename = depth_filename
        self.sem_filename = sem_filename
        self.odom = odom
        self.goal = goal
        self.pair_difficult = pair_difficult
        self.pair_outside   = pair_outside
        self.pair_augment   = pair_augment
        return

    def set_fov(self, fov_angle):
        self.fov_angle = fov_angle
        return

    def __len__(self):
        return len(self.depth_filename)

    def __getitem__(self, idx):
        """
        Get batch items
        
        Returns:
            - depth_image: depth image
            - sem_image: semantic image
            - odom: odometry of the start pose (point and rotation)
            - goal: goal point in the camera frame
            - pair_difficulty: bool if the pair is easy or hard
            - pair_outside: bool if either start or goal is outside the free space
            - pair_augment: bool if the pair is augmented (flipped at the y-axis of the image)
        """
        
        # get depth image
        if self.depth_filename[idx].endswith('.png'):
            depth_image = Image.open(self.depth_filename[idx])
            if self._cfg.real_world_data:
                depth_image = np.array(depth_image.transpose(PIL.Image.ROTATE_180))
            else:
                depth_image = np.array(depth_image)
        else:
            depth_image = np.load(self.depth_filename[idx])
        depth_image[~np.isfinite(depth_image)] = 0.0
        depth_image = (depth_image / 1000.0).astype("float32")
        depth_image[depth_image > self._cfg.max_depth] = 0.0
        # transform depth image
        depth_image = Image.fromarray(depth_image)
        depth_image = self.transform(depth_image)
        if self.pair_augment[idx]:
            depth_image = self.flip_transform.forward(depth_image)  
        
        # get semantic image
        if self.semantics:
            sem_image = Image.open(self.sem_filename[idx])
            if self._cfg.real_world_data:
                sem_image = np.array(sem_image.transpose(PIL.Image.ROTATE_180))
            else:
                sem_image = np.array(sem_image)
            # transform semantic image
            sem_image = Image.fromarray(sem_image)
            sem_image = self.transform(sem_image)
            if self.pair_augment[idx]:
                sem_image = self.flip_transform.forward(sem_image)
        else:
            sem_image = 0  # cannot be None
        
        return depth_image, sem_image, self.odom[idx], self.goal[idx], self.pair_difficult[idx], self.pair_outside[idx], self.pair_augment[idx]


class PlannerDataGenerator(Dataset):
    
    debug = False
    mesh_size = 0.5

    def __init__(
        self, 
        cfg: DataCfg,
        root: str, 
        semantics: bool = False,
        tsdf_map: TSDF_Map = None,
    ) -> None:
        
        # super().__init__()
        # set parameters
        self._cfg = cfg
        self.root = root
        self.tsdf_map = tsdf_map
        self.semantics = semantics 
        
        # init list for final odom, goal and img mapping
        self.depth_filename = []
        self.sem_filename = []
        self.odom_depth: torch.Tensor = None
        self.goal: torch.Tensor = None
        self.pair_outside: np.ndarray = None
        self.pair_difficult: np.ndarray = None
        self.pair_augment: np.ndarray = None
        self.pair_within_fov: np.ndarray = None
        self.pair_front_of_robot: np.ndarray = None
        self.odom_array_sem: pp.LieTensor = None
        self.odom_array_depth: pp.LieTensor = None
        
        self.odom_used: int = 0
        self.odom_no_suitable_goals: int = 0

        # set parameters
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # get odom data and filter
        self.load_odom()
        self.filter_obs_inflation()
        
        # find odom-goal pairs
        self.get_odom_goal_pairs()
        return

    """LOAD HELPER FUNCTIONS"""
    def load_odom(self) -> None:           
        print("Loading odom data...")
        # load odom of every image
        odom_path = os.path.join(self.root, f"camera_extrinsic{self._cfg.depth_suffix}.txt")
        odom_np = np.loadtxt(odom_path, delimiter=",")
        self.odom_array_depth = pp.SE3(odom_np)
        
        if self.semantics:
            odom_path = os.path.join(self.root, f"camera_extrinsic{self._cfg.sem_suffix}.txt")
            odom_np = np.loadtxt(odom_path, delimiter=",")
            self.odom_array_sem = pp.SE3(odom_np)            
        
        if self.debug:
            # plot odom
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size/3.0) # successful trajectory points
            small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
            odom_vis_list = []
            
            for i in range(len(self.odom_array_depth)):
                odom_vis_list.append(copy.deepcopy(small_sphere).translate((self.odom_array_depth[i, 0], self.odom_array_depth[i, 1], self.odom_array_depth[i, 2])))
            odom_vis_list.append(self.tsdf_map.pcd_tsdf)
            
            o3d.visualization.draw_geometries(odom_vis_list)  
        
        return        
    
    def load_images(self, root_path, domain: str = "depth"):
        img_path = os.path.join(root_path, domain)
        assert os.path.isdir(img_path), f"Image directory path '{img_path}' does not exist for domain {domain}"
        assert len(os.listdir(img_path)) > 0, f"Image directory '{img_path}' is empty for domain {domain}"
        
        # use the more precise npy files if available
        img_filename_list = [str(s) for s in Path(img_path).rglob('*.npy')]
        if len(img_filename_list) == 0:
            img_filename_list = [str(s) for s in Path(img_path).rglob('*.png')]
        
        if domain == "depth":
            img_filename_list.sort(key = lambda x : int(x.split("/")[-1][:-(4+len(self._cfg.depth_suffix))]))
        else:
            img_filename_list.sort(key = lambda x : int(x.split("/")[-1][:-(4+len(self._cfg.sem_suffix))]))
        return img_filename_list

    """FILTER HELPER FUNCTIONS"""

    def filter_obs_inflation(self) -> None:
        """
        Filter odom points within the inflation range of the obstacles in the cost map.
        
        Filtering only performed according to the position of the depth camera, due to the close position of depth and semantic camera.
        """
        print("Filter odom points within the inflation range of the obstacles in the cost map...")
        
        norm_inds, _ = self.tsdf_map.Pos2Ind(self.odom_array_depth[:, None, :3])
        cost_grid = self.tsdf_map.cost_array.T.expand(self.odom_array_depth.shape[0], 1, -1, -1)
        norm_inds = norm_inds.to(cost_grid.device)
        oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
        oloss_M = oloss_M.to(torch.float32).to("cpu")
        points_free_space = oloss_M < self._cfg.obs_cost_height
        
        # identify possible start points which have a higher loss value then the one of the intended traversable area
        self.points_outside_free_space_cost = oloss_M > self._cfg.free_space_cost_height
        self.points_outside_free_space_cost = self.points_outside_free_space_cost[points_free_space]

        if self.debug:
            # plot odom
            odom_vis_list = []
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size/3.0) # successful trajectory points

            for i in range(len(self.odom_array_depth)):
                if round(oloss_M[i].item(), 3) == 0.0:
                    small_sphere.paint_uniform_color([0.4, 0.1, 1.0])  # violette
                elif points_free_space[i]:
                    small_sphere.paint_uniform_color([0.4, 1.0, 0.1])  # green
                else:
                    small_sphere.paint_uniform_color([1.0, 0.4, 0.1])  # red
                odom_vis_list.append(copy.deepcopy(small_sphere).translate((self.odom_array_depth[i, 0], self.odom_array_depth[i, 1], self.odom_array_depth[i, 2])))

            odom_vis_list.append(self.tsdf_map.pcd_tsdf)
            o3d.visualization.draw_geometries(odom_vis_list)  

        nb_odom_point_prev = len(self.odom_array_depth)
        self.odom_array_depth = self.odom_array_depth[points_free_space.squeeze()]
        self.nb_odom_points = self.odom_array_depth.shape[0]
        
        if self.semantics:
            self.odom_array_sem = self.odom_array_sem[points_free_space.squeeze()]

        print(f"odom points outside obs inflation : \t{self.nb_odom_points} ({round(self.nb_odom_points/nb_odom_point_prev*100, 2)} %)")
        
        return 
    
    """GENERATE SAMPLES"""
    
    def get_odom_goal_pairs(self) -> None:
        # get fov
        self.get_intrinscs_and_fov()
        # get mesh for raycast check
        self.get_mesh()
        # get pairs
        self.get_pairs()
        
        # free up memory
        self.mesh_wp = None
        self.odom_array_depth = self.odom_array_sem = None
        
        # if percentage of hard or outside samples too small, enable augmentation or repeat them
        ratio_fov, ratio_front, ratio_back, ratio_easy, ratio_hard, ratio_outside = self.compute_ratios()
        self.pair_augment = np.zeros(self.odom_depth.shape[0], dtype=bool)
        
        if ratio_hard > 0.0 and ratio_hard < self._cfg.ratio_fov_hard_samples_max:
            print(f"ratio of hard samples too small ({round(ratio_hard*100, 2)} %), augmenting hard samples")
            
            # expand by all hard samples and activate their mirroring when loaded, mirror goal position
            idx = np.where(self.pair_difficult)[0]
            self.depth_filename         = self.depth_filename + [self.depth_filename[i] for i in idx.tolist()]
            self.sem_filename           = self.sem_filename   + [self.sem_filename[i] for i in idx.tolist()] if self.semantics else None
            self.odom_depth             = torch.vstack([self.odom_depth, self.odom_depth[idx]])
            self.goal                   = torch.vstack([self.goal, self.goal[idx] * torch.tensor([1, -1, 1, 1, 1, 1, 1])])  # include mirror along y axis of the image 
            self.pair_difficult         = np.hstack([self.pair_difficult, self.pair_difficult[idx]])
            self.pair_outside           = np.hstack([self.pair_outside, self.pair_outside[idx]])
            self.pair_augment           = np.hstack([self.pair_augment, np.ones(idx.shape[0], dtype=bool)])
            self.pair_front_of_robot    = np.hstack([self.pair_front_of_robot, self.pair_front_of_robot[idx]])
            
            # calculate new ratios
            ratio_fov, ratio_front, ratio_back, ratio_easy, ratio_hard, ratio_outside = self.compute_ratios()
            print(f"ratio of hard samples increased to ({round(ratio_hard*100, 2)} %)")
            
        # print data mix
        num_easy = sum(self.pair_within_fov) - sum(self.pair_difficult) - sum(self.pair_outside)
        print(
            f"datamix containing {self.odom_depth.shape[0]} suitable odom-goal pairs: \n"
            f"\t fov               : \t{sum(self.pair_within_fov)} ({round(ratio_fov*100, 2)} %) \n"
            f"\t front of robot    : \t{sum(self.pair_front_of_robot)} ({round(ratio_front*100, 2)} %) \n"
            f"\t back of robot     : \t{self.odom_depth.shape[0] - sum(self.pair_within_fov) - sum(self.pair_front_of_robot)} ({round(ratio_back*100, 2)} %) \n"
            f"whereas the fov pairs are distributed as follows: \n"
            f"\t easy pairs        : \t{num_easy} ({round(ratio_easy*100, 2)} %) \n"
            f"\t difficult pairs   : \t{sum(self.pair_difficult)} ({round(ratio_hard*100, 2)} %) \n"
            f"\t outside free space: \t{sum(self.pair_outside)} ({round(ratio_outside*100, 2)} %) \n"
            f"from {self.odom_used} ({round(self.odom_used/self.nb_odom_points*100, 2)} %) different starting points where \n"
            f"\t non-suitable filter: {self.odom_no_suitable_goals} ({round(self.odom_no_suitable_goals/self.nb_odom_points*100, 2)} %)"
        )

        if self.debug:
            start_points =torch.unique(torch.round(self.odom[:, :3], decimals=4), dim=0)
            odom_vis_list = []
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size/3.0) # successful trajectory points

            for i in range(len(start_points)):
                odom_vis_list.append(copy.deepcopy(small_sphere).translate((start_points[i, 0].item(), start_points[i, 1].item(), start_points[i, 2].item())))

            odom_vis_list.append(self.tsdf_map.pcd_tsdf)
            o3d.visualization.draw_geometries(odom_vis_list) 

            goal_points, unique_idx = torch.unique(torch.round(self.goal[:, :3], decimals=4), dim=0, return_inverse=True)
            odom_vis_list = []
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size/3.0) # successful trajectory points

            for i in range(len(goal_points)):
                goal_world = pp.SE3(self.odom[unique_idx[i]]) @ pp.SE3(self.goal[unique_idx[i]])
                odom_vis_list.append(copy.deepcopy(small_sphere).translate((goal_world[0].item(), goal_world[1].item(), goal_world[2].item())))

            odom_vis_list.append(self.tsdf_map.pcd_tsdf)
            o3d.visualization.draw_geometries(odom_vis_list) 
        return
    
    def compute_ratios(self) -> Tuple[float, float, float]:
        # ratio of general samples distribution
        num_within_fov  = self.odom_depth[self.pair_within_fov].shape[0]
        ratio_fov       = num_within_fov/self.odom_depth.shape[0]
        ratio_front     = np.sum(self.pair_front_of_robot)/self.odom_depth.shape[0]
        ratio_back      = (1 - ratio_front - ratio_fov)
        
        # samples ratios within fov samples
        num_easy        = num_within_fov - self.pair_difficult[self.pair_within_fov].sum().item() - self.pair_outside[self.pair_within_fov].sum().item()
        ratio_easy      = num_easy/num_within_fov
        ratio_hard      = self.pair_difficult[self.pair_within_fov].sum().item()/num_within_fov
        ratio_outside   = self.pair_outside[self.pair_within_fov].sum().item()/num_within_fov
        return ratio_fov, ratio_front, ratio_back, ratio_easy, ratio_hard, ratio_outside
    
    def get_intrinscs_and_fov(self) -> None:
        # load intrinsics
        intrinsic_path = os.path.join(self.root, "intrinsics.txt")
        P = np.loadtxt(intrinsic_path, delimiter=",")  # assumes ROS P matrix
        if self.semantics:
            self.K_depth = P[0].reshape(3, 4)[:3, :3]
            self.K_sem = P[1].reshape(3, 4)[:3, :3]
        else:
            self.K_depth = P.reshape(3, 4)[:3, :3]
            
        self.alpha_fov = 2 * math.atan(self.K_depth[0, 0] / self.K_depth[0, 2])
        return
    
    def get_mesh(self) -> None:
        # transform cost map to nvidia warp mesh
        print("Transforming cost map to mesh...")
        if os.path.isfile(os.path.join(self.root, "cost_mesh.ply")):
            cost_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root, "cost_mesh.ply"))
        else:
            # estimate normals of pcd
            self.tsdf_map.pcd_tsdf.estimate_normals()
            cost_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.tsdf_map.pcd_tsdf , depth=12)    
            o3d.io.write_triangle_mesh(os.path.join(self.root, "cost_mesh.ply"), cost_mesh)

        self.mesh_wp = wp.Mesh(
            points=wp.array(np.asarray(cost_mesh.vertices).astype(np.float32), dtype=wp.vec3, device=self._device),
            indices=wp.array(np.asarray(cost_mesh.triangles).astype(np.int32).flatten(), dtype=int, device=self._device),
        )
        print("Done.")
        return

    def get_pairs(self):
        print("Generating pairs of start and end points ...")
        # load depth image files as name list
        depth_filename_list = self.load_images(self.root, "depth")
        sem_filename_list = self.load_images(self.root, "semantics") if self.semantics else None 

        # iterate over all odom points and find goal points
        self.odom_no_suitable_goals = 0
        self.odom_used = 0
        
        # init buffer lists
        odom_list_dep = []
        goal_list = []
        pair_outside = []
        pair_within_fov = []
        pair_front_of_robot = []
        pair_difficult = []
        
        # init semantic warp parameters
        if self.semantics:
            # get intrinsics of both cameras
            K_depth, K_sem = self._get_intrinsics()
            # compute pixel tensor
            pix_depth_cam_frame = self._compute_pixel_tensor(K_depth, depth_filename=depth_filename_list[0])
            # make dir
            os.makedirs(os.path.join(self.root, "semantics_warp"), exist_ok=True)
            
        # iterate over all odom points
        for odom_idx in tqdm(range(self.nb_odom_points), desc="odom points"):
            odom = self.odom_array_depth[odom_idx]
            
            # find goal in robot fov
            goals, outside_free_space_cost, within_fov, front_of_robot = self.find_goal(odom)  # returns goals in odom frame
            if len(goals) == 0:
                self.odom_no_suitable_goals += 1
                continue
            self.odom_used += 1
            
            # evaluate goal difficulty for odom-goal paris within the fov
            pair_difficult_current = torch.zeros(len(goals), dtype=torch.bool)
            pair_difficult_current[within_fov] = self.get_goal_difficulty(odom, goals[within_fov])
            
            # reduce number of goals behind the robot
            behind_robot = ~front_of_robot.clone()
            behind_robot[within_fov] = False
            idx_behind = self.reduce_pairs(behind_robot)

            # reduce number of goals in front of the robot but outside fov
            idx_front = self.reduce_pairs(front_of_robot)
            
            # reduce easy samples within fov but keep all hard ones
            fov_non_difficult = within_fov.clone()
            fov_non_difficult[pair_difficult_current] = False
            idx_difficult = self.reduce_pairs(pair_difficult_current)
            idx_fov = self.reduce_pairs(fov_non_difficult)
                
            # combine reduction indices
            idx_all = torch.hstack((idx_behind, idx_front, idx_difficult, idx_fov))            
            goals = goals[idx_all]
            outside_free_space_cost = outside_free_space_cost[idx_all]
            within_fov = within_fov[idx_all]
            front_of_robot = front_of_robot[idx_all]
            pair_difficult_current = pair_difficult_current[idx_all]
            
            # process found categroies of odom-goal pairs
            pair_within_fov.append(within_fov.numpy())
            pair_front_of_robot.append(front_of_robot.numpy())
            pair_difficult.append(pair_difficult_current.numpy())
            pair_outside.append(torch.any(
                torch.vstack((outside_free_space_cost, self.points_outside_free_space_cost[odom_idx].repeat(outside_free_space_cost.shape[0]))), dim=0
            ).numpy())
            # process odom and goal
            odom_list_dep.append(odom.repeat(len(goals), 1))
            goal_list.append(goals)
            # process depth and semantic images
            self.depth_filename += [depth_filename_list[odom_idx]] * len(goals)
            if self.semantics:
                # semantic warp
                sem_img_new_path = self._get_overlay_semantics(
                    depth_filename=depth_filename_list[odom_idx],
                    sem_filename=sem_filename_list[odom_idx],
                    K_sem=K_sem,
                    pose_dep=odom,
                    pose_sem=self.odom_array_sem[odom_idx],
                    pix_depth_cam_frame=pix_depth_cam_frame
                )
                self.sem_filename += [sem_img_new_path] * len(goals)
            
            # DEBUG
            if self.debug:
                # plot odom
                small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size/3.0) # successful trajectory points
                odom_vis_list = []
                goal_odom = odom @ goals
                hit_pcd = (goal_odom).cpu().numpy()[:, :3]
                for idx, pts in enumerate(hit_pcd):
                    if within_fov[idx]:
                        small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
                    elif front_of_robot[idx]:
                        small_sphere.paint_uniform_color([0.0, 0.5, 0.5])
                    else:
                        small_sphere.paint_uniform_color([0.0, 0.1, 1.0])
                    odom_vis_list.append(copy.deepcopy(small_sphere).translate((pts[0], pts[1], pts[2])))
                
                # mesh viz
                cost_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root, "cost_mesh.ply"))
                cost_mesh.compute_vertex_normals()
                cost_mesh.paint_uniform_color([0.4, 0.4, 0.4])
                odom_vis_list.append(cost_mesh)
                
                # field of view visualization 
                fov_vis_length = 0.75  # length of the fov visualization plane in meters
                fov_vis_pt_right = odom @ pp.SE3([fov_vis_length * np.cos(self.alpha_fov/2), fov_vis_length * np.sin(self.alpha_fov/2), 0, 0, 0, 0, 1])
                fov_vis_pt_left = odom @ pp.SE3([fov_vis_length * np.cos(self.alpha_fov/2), -fov_vis_length * np.sin(self.alpha_fov/2), 0, 0, 0, 0, 1])
                fov_vis_pt_right = fov_vis_pt_right.numpy()[:3]
                fov_vis_pt_left = fov_vis_pt_left.numpy()[:3]
                fov_mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(np.array([odom.data.cpu().numpy()[:3], fov_vis_pt_right, fov_vis_pt_left])),
                    triangles=o3d.utility.Vector3iVector(np.array([[2, 1, 0]]))
                )
                fov_mesh.paint_uniform_color([1.0, 0.5, 0.0])
                odom_vis_list.append(fov_mesh)
                
                # odom viz
                small_sphere.paint_uniform_color([1.0, 0.0, 0.0])            
                odom_vis_list.append(copy.deepcopy(small_sphere).translate((odom.data[0].item(), odom.data[1].item(), odom.data[2].item())))

                # plot goal
                o3d.visualization.draw_geometries(odom_vis_list)
                
        # transform lists to entire tensors and arrays
        self.odom_depth             = torch.vstack(odom_list_dep).data
        self.goal                   = torch.vstack(goal_list).data
        self.pair_outside           = np.hstack(pair_outside)
        self.pair_within_fov        = np.hstack(pair_within_fov)
        self.pair_front_of_robot    = np.hstack(pair_front_of_robot)
        self.pair_difficult         = np.hstack(pair_difficult)
        
        return

    def reduce_pairs(self, decision_tensor) -> torch.Tensor:
        # indicies to keep
        indices = torch.randperm(torch.sum(decision_tensor))
        if len(indices) > self._cfg.max_goal_per_odom:
            indices = indices[:self._cfg.max_goal_per_odom] 
        return torch.nonzero(decision_tensor).flatten()[indices]
        
    def get_goal_difficulty(self, odom: pp.LieTensor, goals: pp.LieTensor):
        """evaluate if goals within the fov are hard to reach (goals outside fov are always hard to reach)"""
        # filter odom-goal pairs in free space
        
        # get start and end of ray if both are in free space (i.e. not with a cost value over cfg.free_space_cost_height)
        ray_sta = odom[:3].data.to(self.tsdf_map.device)
        ray_end = (odom @ goals).data.to(self.tsdf_map.device)[:, :3]
        
        # get ray direction (normalize it), it's maximum length and assign the z values of the rays
        ray_dir = ray_end - ray_sta
        ray_dir[:, 2] = 0
        ray_sta[2] = self._cfg.obs_cost_height
        ray_len = torch.norm(ray_dir, dim=1)
        ray_dir = ray_dir / ray_len[:, None]
        
        # raycast -> check if obstacle between start and end
        _, ray_hit_depth = _raycast(
            self.mesh_wp,
            ray_starts_world=ray_sta.repeat(ray_dir.shape[0], 1),
            ray_directions_world=ray_dir,
            max_depth=ray_len,
        )
        
        return torch.isfinite(ray_hit_depth).cpu()
            
    def find_goal(self, odom):
        """
        Make sure that the goal is in the robot's fov, otherwise find a new goal, maximum iterate over max_iter points
        """
        # get all odom point within current odom frame
        goal_odom_frame = (pp.Inv(odom) @ self.odom_array_depth)
        
        # max distance between goal and odom
        goal_cam_sqrt = torch.sqrt(torch.sum((goal_odom_frame**2)[:, :2], dim=1))
        idx_max = goal_cam_sqrt < self._cfg.max_goal_distance
        idx_min = goal_cam_sqrt > self._cfg.min_goal_distance
        within_distance = torch.all(torch.stack([idx_max, idx_min]), dim=0)
        goal_within_distance = goal_odom_frame[within_distance]
        outside_free_space_cost = self.points_outside_free_space_cost[within_distance]
        
        # check for each goal if there is a wall blocking the way between odom and goal
        goal_world_frame = self.odom_array_depth[within_distance]
        orientations = goal_world_frame.data[:, :3] - odom[:3]
        norms = torch.norm(orientations, dim=1)
        without_wall = self.check_for_wall(odom, orientations/norms[:, None], norms)
        goal_without_wall = goal_within_distance[without_wall]
        outside_free_space_cost = outside_free_space_cost[without_wall]

        # get if odom-goal is within fov or outside the fov but still in front of the robot
        goal_angle = abs(torch.atan2(goal_without_wall.data[:, 1], goal_without_wall.data[:, 0]))
        within_fov = goal_angle < self.alpha_fov/2 * self._cfg.fov_scale
        front_of_robot = goal_angle < torch.pi/2
        front_of_robot[within_fov] = False
        
        if self.debug:
            # plot odom
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size/3.0) # successful trajectory points
            odom_vis_list = []
            within_distance_odom = self.odom_array_depth.data[within_distance, :3]
            without_wall_odom = within_distance_odom[without_wall]
            hit_pcd = (without_wall_odom).cpu().numpy()
            for idx, pts in enumerate(hit_pcd):
                if within_fov[idx]:
                    small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
                elif front_of_robot[idx]:
                    small_sphere.paint_uniform_color([0.0, 0.5, 0.5])
                else:
                    small_sphere.paint_uniform_color([0.0, 0.1, 1.0])
                odom_vis_list.append(copy.deepcopy(small_sphere).translate((pts[0], pts[1], pts[2])))
            
            # mesh viz
            cost_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root, "cost_mesh.ply"))
            cost_mesh.compute_vertex_normals()
            cost_mesh.paint_uniform_color([0.4, 0.4, 0.4])
            odom_vis_list.append(cost_mesh)
            
            # field of view visualization 
            fov_vis_length = 0.75  # length of the fov visualization plane in meters
            fov_vis_pt_right = odom @ pp.SE3([fov_vis_length * np.cos(self.alpha_fov/2), fov_vis_length * np.sin(self.alpha_fov/2), 0, 0, 0, 0, 1])
            fov_vis_pt_left = odom @ pp.SE3([fov_vis_length * np.cos(self.alpha_fov/2), -fov_vis_length * np.sin(self.alpha_fov/2), 0, 0, 0, 0, 1])
            fov_vis_pt_right = fov_vis_pt_right.numpy()[:3]
            fov_vis_pt_left = fov_vis_pt_left.numpy()[:3]
            fov_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(np.array([odom.data.cpu().numpy()[:3], fov_vis_pt_right, fov_vis_pt_left])),
                triangles=o3d.utility.Vector3iVector(np.array([[2, 1, 0]]))
            )
            fov_mesh.paint_uniform_color([1.0, 0.5, 0.0])
            odom_vis_list.append(fov_mesh)
            
            # odom viz
            small_sphere.paint_uniform_color([1.0, 0.0, 0.0])            
            odom_vis_list.append(copy.deepcopy(small_sphere).translate((odom.data[0].item(), odom.data[1].item(), odom.data[2].item())))

            # plot goal
            o3d.visualization.draw_geometries(odom_vis_list)
        
        return goal_without_wall, outside_free_space_cost, within_fov, front_of_robot

    def check_for_wall(self, odom, orientations, goal_distances) -> float:
        """ Determine if there is a wall blocking the whole fov between the start and goal point, if goal, give minumum distance to wall"""
        angles = np.linspace(-self.alpha_fov/2, self.alpha_fov/2, self._cfg.n_rays_check)
        ray_directions = np.zeros((self._cfg.n_rays_check * orientations.shape[0], 3))
        ray_distances = torch.zeros(self._cfg.n_rays_check * orientations.shape[0], device=self.tsdf_map.device, dtype=torch.float32)
        for i in range(orientations.shape[0]):
            ray_directions[i*self._cfg.n_rays_check:(i+1)*self._cfg.n_rays_check, :] = tf.Rotation.from_euler('z', angles, degrees=False).as_matrix() @ orientations[i].numpy()
            ray_distances[i*self._cfg.n_rays_check:(i+1)*self._cfg.n_rays_check] = goal_distances[i]
        ray_directions[:, -1] = 0  # tilt of camera not regarded to avoid recognizing bottom as obstacles
        ray_directions = torch.tensor(ray_directions, device=self.tsdf_map.device, dtype=torch.float32)
        ray_origin = odom.data[:3].repeat(ray_directions.shape[0], 1).to(self.tsdf_map.device)
        
        _, ray_hit_depth = _raycast(
            self.mesh_wp,
            ray_starts_world=ray_origin,
            ray_directions_world=ray_directions,
            max_depth=ray_distances,
        )
        
        without_wall = torch.zeros(orientations.shape[0], dtype=torch.bool)
        for i in range(orientations.shape[0]):
            without_wall[i] = torch.sum(torch.isfinite(ray_hit_depth[i*self._cfg.n_rays_check:(i+1)*self._cfg.n_rays_check]))/self._cfg.n_rays_check < self._cfg.ray_obs_ratio
    
            if self.debug:
                # plot odom
                small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size/3.0) # successful trajectory points
                small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
                odom_vis_list = []
                hit_pcd = (ray_origin[i*self._cfg.n_rays_check:(i+1)*self._cfg.n_rays_check] + ray_hit_depth[i*self._cfg.n_rays_check:(i+1)*self._cfg.n_rays_check, None] * ray_directions[i*self._cfg.n_rays_check:(i+1)*self._cfg.n_rays_check]).cpu().numpy()
                hit_pcd = hit_pcd[~np.any(np.isnan(hit_pcd), axis=1)]
                
                for pts in hit_pcd:
                    odom_vis_list.append(copy.deepcopy(small_sphere).translate((pts[0], pts[1], pts[2])))

                # mesh viz            
                cost_mesh = o3d.io.read_triangle_mesh(os.path.join(self.root, "cost_mesh.ply"))
                cost_mesh.compute_vertex_normals()
                cost_mesh.paint_uniform_color([0.4, 0.4, 0.4])
                odom_vis_list.append(cost_mesh)
                
                # odom viz
                small_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                odom_vis_list.append(copy.deepcopy(small_sphere).translate((odom.data.cpu().numpy()[0], odom.data.cpu().numpy()[1], odom.data.cpu().numpy()[2])))

                # goal viz
                goal = odom.data[:3] + goal_distances[i] * orientations[i]
                small_sphere.paint_uniform_color([0.0, 0.0, 1.0])
                odom_vis_list.append(copy.deepcopy(small_sphere).translate((goal.data.cpu().numpy()[0], goal.data.cpu().numpy()[1], goal.data.cpu().numpy()[2])))
                
                # plot goal
                o3d.visualization.draw_geometries(odom_vis_list)

        return without_wall
    
    """SPLIT HELPER FUNCTIONS"""

    def split_samples(
        self,
        test_dataset: PlannerData,
        train_dataset: Optional[PlannerData] = None,
        generate_split: bool = False,
        ratio_fov_samples: Optional[float] = None,
        ratio_front_samples: Optional[float] = None,
        ratio_back_samples: Optional[float] = None,
        ratio_fov_hard_samples: Optional[float] = None,
        ratio_fov_easy_samples: Optional[float] = None,
        ratio_fov_outside_samples: Optional[float] = None,
    ) -> None:
        
        # checkk if ratios are given or defaults are used 
        ratio_fov_samples           = ratio_fov_samples if ratio_fov_samples is not None else self._cfg.ratio_fov_samples
        ratio_front_samples         = ratio_front_samples if ratio_front_samples is not None else self._cfg.ratio_front_samples
        ratio_back_samples          = ratio_back_samples if ratio_back_samples is not None else self._cfg.ratio_back_samples
        ratio_fov_easy_samples      = ratio_fov_easy_samples if ratio_fov_easy_samples is not None else self._cfg.ratio_fov_easy_samples
        ratio_fov_hard_samples      = ratio_fov_hard_samples if ratio_fov_hard_samples is not None else self._cfg.ratio_fov_hard_samples
        ratio_fov_outside_samples   = ratio_fov_outside_samples if ratio_fov_outside_samples is not None else self._cfg.ratio_fov_outside_samples
        
        assert round(ratio_fov_samples + ratio_front_samples + ratio_back_samples, 2) == 1.0, f"Sample ratios must sum up to 1.0, currently {ratio_back_samples + ratio_front_samples + ratio_fov_samples}"
        assert round(ratio_fov_easy_samples + ratio_fov_hard_samples + ratio_fov_outside_samples, 2) == 1.0, f"FOV sample ratios must sum up to 1.0, currently {ratio_fov_easy_samples + ratio_fov_hard_samples + ratio_fov_outside_samples}"
        
        # get current idx
        idx = np.arange(self.odom_depth.shape[0])
        idx_fov = idx[self.pair_within_fov]
        idx_front = idx[self.pair_front_of_robot]
        idx_back = np.delete(idx, np.concatenate((idx_fov, idx_front)))
        idx_fov_hard = idx[self.pair_difficult]
        idx_fov_outside = idx[self.pair_outside]
        idx_fov_easy = np.delete(idx, np.concatenate((idx_back, idx_front, idx_fov_hard, idx_fov_outside)))

        # select indexes
        max_sample_number = int(self._cfg.max_train_pairs / self._cfg.ratio)
        
        idx_front       = np.random.choice(idx_front,       int(max_sample_number * self._cfg.ratio_front_samples),       replace=False if len(idx_front)       > int(max_sample_number * self._cfg.ratio_front_samples)       else True)
        idx_back        = np.random.choice(idx_back,        int(max_sample_number * self._cfg.ratio_back_samples),        replace=False if len(idx_back)        > int(max_sample_number * self._cfg.ratio_back_samples)        else True)
        idx_fov_easy    = np.random.choice(idx_fov_easy,    int(max_sample_number * self._cfg.ratio_fov_easy_samples),    replace=False if len(idx_fov_easy)    > int(max_sample_number * self._cfg.ratio_fov_easy_samples)    else True)
        idx_fov_hard    = np.random.choice(idx_fov_hard,    int(max_sample_number * self._cfg.ratio_fov_hard_samples),    replace=False if len(idx_fov_hard)    > int(max_sample_number * self._cfg.ratio_fov_hard_samples)    else True)
        idx_fov_outside = np.random.choice(idx_fov_outside, int(max_sample_number * self._cfg.ratio_fov_outside_samples), replace=False if len(idx_fov_outside) > int(max_sample_number * self._cfg.ratio_fov_outside_samples) else True)
        idx_selected = np.concatenate((idx_front, idx_back, idx_fov_easy, idx_fov_hard, idx_fov_outside))
        
        # print final training mix
        print(f"Generated {self._cfg.max_train_pairs} odom-goal pairs to train and {round(max_sample_number - self._cfg.max_train_pairs)} validation pairs from {self.odom_depth.shape[0]} pairs")
        print(f"Selected data-mix: \n"
              f"\t front of robot    : \t{len(idx_front)}\t({round(self._cfg.ratio_front_samples*100, 2)} %) \n"
              f"\t back of robot     : \t{len(idx_back)}\t({round(self._cfg.ratio_back_samples*100, 2)} %) \n"
              f"\t fov samples       : \t{len(idx_fov_easy)+len(idx_fov_hard)+len(idx_fov_outside)}\t({round(self._cfg.ratio_fov_samples*100, 2)} %) \n"
              f"Whereas fov samples are sparated as follows: \n"
              f"\t easy pairs        : \t{len(idx_fov_easy)}\t({round(self._cfg.ratio_fov_easy_samples*100, 2)} %) \n"
              f"\t difficult pairs   : \t{len(idx_fov_hard)}\t({round(self._cfg.ratio_fov_hard_samples*100, 2)} %) \n"
              f"\t outside free space: \t{len(idx_fov_outside)}s\t({round(self._cfg.ratio_fov_outside_samples*100, 2)} %) \n"
              f"from {torch.unique(self.odom_depth, dim=1).shape[0]} different starting points")

        # generate split
        if generate_split:
            idx = np.arange(len(idx_selected))
            sampled_idx = sample(idx.tolist(), int(len(idx_selected) * self._cfg.ratio))
            train_index = idx_selected[sampled_idx]
            idx_selected = np.delete(idx_selected, sampled_idx)
        
            train_dataset.update_buffers(
                depth_filename  = [self.depth_filename[i] for i in train_index],
                sem_filename    = [self.sem_filename[i] for i in train_index] if self.semantics else None,
                odom            = self.odom_depth[train_index],
                goal            = self.goal[train_index],
                pair_difficult  = self.pair_difficult[train_index],
                pair_outside    = self.pair_outside[train_index],
                pair_augment    = self.pair_augment[train_index]
            )
            train_dataset.set_fov(self.alpha_fov)    

        test_dataset.update_buffers(
            depth_filename  = [self.depth_filename[i] for i in idx_selected],
            sem_filename    = [self.sem_filename[i] for i in idx_selected] if self.semantics else None,
            odom            = self.odom_depth[idx_selected],
            goal            = self.goal[idx_selected],
            pair_difficult  = self.pair_difficult[idx_selected],
            pair_outside    = self.pair_outside[idx_selected],
            pair_augment    = self.pair_augment[idx_selected] 
        )
        test_dataset.set_fov(self.alpha_fov)

        return
    
    """ Warp semantic on depth image helper functions"""
        
    def _get_intrinsics(self) -> None:
        intrinsic_path = os.path.join(self.root, "intrinsics.txt")
        P = np.loadtxt(intrinsic_path, delimiter=",")  # assumes ROS P matrix
        K_depth = P[0].reshape(3, 4)[:3, :3]
        K_sem = P[1].reshape(3, 4)[:3, :3]
        return K_depth, K_sem
        
    def _compute_pixel_tensor(self, K_depth, depth_filename) -> None:
        # get shape of depth images
        depth_img = self._load_depth_image(depth_filename)
        x_nums, y_nums = depth_img.shape
        
        # get image plane mesh grid
        pix_u = np.arange(0, y_nums)
        pix_v = np.arange(0, x_nums)
        grid = np.meshgrid(pix_u, pix_v)
        pixels = np.vstack(list(map(np.ravel, grid))).T
        pixels = np.hstack(
            [pixels, np.ones((len(pixels), 1))]
        )  # add ones for 3D coordinates           
        
        # transform to camera frame
        k_inv = np.linalg.inv(K_depth)
        pix_cam_frame = np.matmul(k_inv, pixels.T)
        # reorder to be in "robotics" axis order (x forward, y left, z up)
        return pix_cam_frame[[2, 0, 1], :].T  * np.array([1, -1, -1])
    
    def _load_depth_image(self, depth_filename):
        if depth_filename.endswith('.png'):
            depth_image = Image.open(depth_filename)
            if self._cfg.real_world_data:
                depth_image = np.array(depth_image.transpose(PIL.Image.ROTATE_180))
            else:
                depth_image = np.array(depth_image)
        else:
            depth_image = np.load(depth_filename)
        
        depth_image[~np.isfinite(depth_image)] = 0.0
        depth_image = (depth_image / self._cfg.depth_scale).astype("float32")
        depth_image[depth_image > self._cfg.max_depth] = 0.0
        return depth_image
    
    def _get_overlay_semantics(self, depth_filename, sem_filename, K_sem, pose_sem, pose_dep, pix_depth_cam_frame):
        # load semantic and depth image and get their poses
        depth_img = self._load_depth_image(depth_filename)
        sem_image = Image.open(sem_filename)
        if self._cfg.real_world_data:
            sem_image = np.array(sem_image.transpose(PIL.Image.ROTATE_180))
        else:
            sem_image = np.array(sem_image)
        pose_dep = pose_dep.data.cpu().numpy()
        pose_sem = pose_sem.data.cpu().numpy()
        
        # get 3D points of depth image
        rot = tf.Rotation.from_quat(pose_dep[3:]).as_matrix()
        dep_im_reshaped = depth_img.reshape(-1, 1) # flip s.t. start in lower left corner of image as (0,0) -> has to fit to the pixel tensor
        points = dep_im_reshaped * (rot @ pix_depth_cam_frame.T).T + pose_dep[:3]
        
        # transform points to semantic camera frame
        points_sem_cam_frame = (tf.Rotation.from_quat(pose_sem[3:]).as_matrix().T @ (points - pose_sem[:3]).T).T
        # normalize points
        points_sem_cam_frame_norm = points_sem_cam_frame / points_sem_cam_frame[:, 0][:, np.newaxis]
        # reorder points be camera convention (z-forward)
        points_sem_cam_frame_norm = points_sem_cam_frame_norm[:, [1, 2, 0]]  * np.array([-1, -1, 1])
        # transform points to pixel coordinates
        pixels = (K_sem @ points_sem_cam_frame_norm.T).T
        # filter points outside of image
        filter_idx = (pixels[:, 0] >= 0) & (pixels[:, 0] < sem_image.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < sem_image.shape[0])
        # get semantic annotation
        sem_annotation = np.zeros((pixels.shape[0], 3))
        sem_annotation[filter_idx] = sem_image[pixels[filter_idx, 1].astype(int), pixels[filter_idx, 0].astype(int)]
        # reshape to image
        
        sem_image_warped = sem_annotation.reshape(depth_img.shape[0], depth_img.shape[1], 3)
        
        # DEBUG
        if self.debug:
            import matplotlib.pyplot as plt
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(depth_img)
            ax2.imshow(sem_image_warped / 255)
            ax3.imshow(depth_img)
            ax3.imshow(sem_image_warped / 255, alpha=0.5)
            plt.show()
        
        # save semantic image under the new path
        sem_filename = os.path.split(sem_filename)[1]
        sem_image_path = os.path.join(self.root, "semantics_warp", sem_filename)
        cv2.imwrite(sem_image_path, sem_image_warped)
        
        return sem_image_path

# EoF
