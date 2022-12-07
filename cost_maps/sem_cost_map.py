#!/usr/bin python3

"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      semantic cost map for imperative learning
"""

# python
import os
import open3d as o3d
import numpy as np
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Tuple

# imperative-cost-map
from config import SemCostMapConfig, GeneralCostMapConfig, CARLA_LOSS, MATTERPORT_LOSS, OBSTACLE_LOSS


class SemCostMap:
    """
    Cost Map based on semantic information
    """
    
    def __init__(self, cfg_general: GeneralCostMapConfig, cfg: SemCostMapConfig, visualize: bool = True):
        self._cfg_general = cfg_general
        self._cfg_sem = cfg
        self.visualize = visualize
        
        # cost map init parameters
        self.pcd: o3d.geometry.PointCloud = None
        self.pcd_filtered: o3d.geometry.PointCloud = None
        self._num_x: int = None
        self._num_y: int = None
        self._start_x: float = None
        self._start_y: float = None
        self._init_done: bool = False
        
        # cost map
        self.grid_cell_loss: np.ndarray = None
        return
    
    
    def pcd_init(self) -> None:
        # load pcd and filter it
        print("COST-MAP INIT START")
        print(f"start loading and filtering point cloud from: {self._cfg_general.ply_file}")
        self.pcd = o3d.io.read_point_cloud(os.path.join(self._cfg_general.root_path, self._cfg_general.ply_file))
        self.pcd_filtered = self._pcd_filter()
        
        # update init flag
        self._init_done = True
        print("COST-MAP INIT DONE")
        return

    def create_costmap(self) -> Tuple[list, list]:
        assert self._init_done, "cost map not initialized, call pcd_init() first"
        print("COST-MAP CREATION START")
        
        # get class idx (=class) of each point in the point cloud
        if self._cfg_sem.data_source == "matterport":
            class_idx = self._mapping_matterport()
            class_loss = list(MATTERPORT_LOSS.values())
        elif self._cfg_sem.data_source == "carla":
            raise NotImplementedError
            class_loss = list(CARLA_LOSS.values())
        else:
            raise ValueError(f"unknown data source: {self._cfg_sem.data_source}")
        
        # update map parameters --> has to be done after mapping because last step where points are removed
        self._set_map_parameters()
        
        # assign each point its loss
        pts_loss = np.zeros(class_idx.shape[0])
        for sem_class in range(len(class_loss)):
            pts_loss[class_idx == sem_class] = class_loss[sem_class]
        
        # get the loss for each grid cell
        self.grid_cell_loss = self._get_grid_loss(pts_loss)

        # smooth cost map
        # grid_cell_loss = ndimage.distance_transform_edt(grid_cell_loss)
        # grid_cell_loss[grid_cell_loss > 0.0]  = np.log(grid_cell_loss[grid_cell_loss > 0.0] + math.e)
        # self.grid_cell_loss = ndimage.gaussian_filter(grid_cell_loss, sigma=self._cfg_general.sigma_smooth)

        print("COST-MAP CREATION DONE")
        
        # TODO: Change when using true terrain analysis module to make applicable for non-flat surfaces
        ground_array = np.ones([self._num_x, self._num_y]) * 0.0
        return [self.grid_cell_loss, self.pcd_filtered.points, ground_array], [self._start_x, self._start_y]

    def viz_costmap(self):
        plt.imshow(self.grid_cell_loss, cmap="jet")
        plt.show()
        return
            
    """Helper functions"""

    def _pcd_filter(self) -> o3d.geometry.PointCloud:
        """remove points above the robot height and filter for outliers"""
        
        # remove points above the robot height 
        # TODO: take ground height into account --> currently only for flat surfaces
        pts = np.asarray(self.pcd.points)
        pts_ceil_idx = pts[:, 2] < self._cfg_sem.robot_height * self._cfg_sem.robot_height_factor
        pts_ground_idx = pts[:, 2] > self._cfg_sem.ground_height
        pcd_height_filtered = self.pcd.select_by_index(np.where(np.vstack((pts_ceil_idx, pts_ground_idx)).all(axis=0))[0])
        
        # remove statistical outliers
        pcd_filtered, _ = pcd_height_filtered.remove_statistical_outlier(nb_neighbors=self._cfg_sem.nb_neighbors, std_ratio=self._cfg_sem.std_ratio)
        
        return pcd_filtered
    
    def _set_map_parameters(self) -> None:
        """Define the size and start position of the cost map"""
        pts = np.asarray(self.pcd_filtered.points)
        assert pts.shape[0] > 0, "No points received."
        
        # get max and minimum of cost map
        max_x, max_y, _ = np.round(np.amax(pts, axis=0), decimals=1) + self._cfg_general.clear_dist
        min_x, min_y, _ = np.round(np.amin(pts, axis=0), decimals=1) - self._cfg_general.clear_dist

        self._num_x = np.ceil((max_x - min_x) / self._cfg_general.resolution).astype(int)
        self._num_y = np.ceil((max_y - min_y) / self._cfg_general.resolution).astype(int)
        self._start_x = (max_x + min_x) / 2.0 - self._num_x / 2.0 * self._cfg_general.resolution
        self._start_y = (max_y + min_y) / 2.0 - self._num_y / 2.0 * self._cfg_general.resolution
        print(f"cost map size set to: {self._num_x} x {self._num_y}")
        return

    def _mapping_matterport(self) -> np.ndarray:
        """mapping between color and matterport semantic classes"""
        # get colors
        color = np.asarray(self.pcd_filtered.colors) * 255.0
        
        # load defined colors for mpcat40
        mapping_40 = pd.read_csv(self._cfg_sem.mapping_dir + "/mpcat40.tsv", sep='\t')
        color_to_id = mapping_40["hex"].to_numpy()
        color_to_id = np.array([(int(color_to_id[i][1:3], 16), int(color_to_id[i][3:5], 16), int(color_to_id[i][5:7], 16)) for i in range(len(color_to_id))])
        
        # pts to class idx array
        pts_class_idx = np.ones(color.shape[0], dtype=int) * -1

        # assign each point to a class
        color = color.astype(int)
        for class_idx, class_color in enumerate(color_to_id):
            pts_idx_of_class = (color == class_color).all(axis=1).nonzero()[0]
            pts_class_idx[pts_idx_of_class] = class_idx
        
        # identify points with unknown classes --> remove from point cloud
        known_idx = np.where(pts_class_idx != -1)[0]
        self.pcd_filtered = self.pcd_filtered.select_by_index(known_idx)
        print(f"Class of {len(known_idx)} points identified ({len(known_idx) / len(color)} %).")

        return pts_class_idx[known_idx]
    
    def _get_grid_loss(self, pts_loss: np.ndarray) -> np.ndarray:
        """convert points to grid"""
        # get points
        pts = np.asarray(self.pcd_filtered.points)
        nb_pts = pts.shape[0]
        
        # init obstacle map
        grid_obs = np.zeros([self._num_x, self._num_y])
        grid_cell_count = np.zeros([self._num_x, self._num_y])
        
        while True:
            # grid idx 
            pts_grid_idx = (np.round((pts[:, :2] - np.array([self._start_x, self._start_y])) / self._cfg_general.resolution)).astype(int)

            # update loss and counter
            grid_idx, pts_to_grid_idx_map = np.unique(pts_grid_idx, axis=0, return_index=True)
            grid_obs[grid_idx[:,0], grid_idx[:,1]] += pts_loss[pts_to_grid_idx_map]
            grid_cell_count[grid_idx[:,0], grid_idx[:,1]] += 1
            
            # remove processed points
            pts = np.delete(pts, pts_to_grid_idx_map, axis=0)
            pts_loss = np.delete(pts_loss, pts_to_grid_idx_map, axis=0)
            
            # terminate when all points are processed
            if len(pts) == 0:
                break
        
        assert np.sum(grid_cell_count) == nb_pts, "Not all points are processed."
        
        # average loss per grid cell
        grid_obs[np.nonzero(grid_cell_count)] /= grid_cell_count[np.nonzero(grid_cell_count)]
        
        # create cost map out of it
        grid_cell_loss = np.ones([self._num_x, self._num_y]) * OBSTACLE_LOSS
        grid_cell_loss[grid_cell_count.nonzero()] = 0.0  # get free space map
        grid_cell_loss += grid_obs
        
        
        
        
        grid_cell_loss = ndimage.grey_erosion(grid_cell_loss, size=(3,3))
        
        # increase obstacle and free space
        grid_obs = ndimage.gaussian_filter(grid_obs, sigma=self._cfg_general.sigma_expand*2)        
        grid_cell_loss = ndimage.gaussian_filter(grid_cell_loss, sigma=self._cfg_general.sigma_expand_free*2)

        # apply thresholds
        grid_cell_loss[grid_cell_loss < self._cfg_general.free_space_threshold] = 0.0  # assign free space
        grid_cell_loss[grid_obs > self._cfg_general.obstacle_threshold] = 1.0  # assign obstacles
        
        if self.visualize:
            plt.imshow(grid_cell_loss)
            plt.show()
            
        return grid_cell_loss


# EoF
