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
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Tuple
from sklearn.neighbors import KNeighborsRegressor
import multiprocessing as mp
from functools import partial
import alphashape

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
        
        # get the loss for each grid cell
        grid_loss = self._get_grid_loss(class_idx=class_idx, class_loss=class_loss)
        
        # make grid loss differentiable
        grid_loss = self._dense_grid_loss(grid_loss)

        print("COST-MAP CREATION DONE")
        
        # TODO: Change when using true terrain analysis module to make applicable for non-flat surfaces
        ground_array = np.ones([self._num_x, self._num_y]) * 0.0
        return [grid_loss, self.pcd_filtered.points, ground_array], [self._start_x, self._start_y]

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
        self.color_to_id = np.array([(int(color_to_id[i][1:3], 16), int(color_to_id[i][3:5], 16), int(color_to_id[i][5:7], 16)) for i in range(len(color_to_id))])
        
        # pts to class idx array
        pts_class_idx = np.ones(color.shape[0], dtype=int) * -1

        # assign each point to a class
        color = color.astype(int)
        for class_idx, class_color in enumerate(self.color_to_id):
            pts_idx_of_class = (color == class_color).all(axis=1).nonzero()[0]
            pts_class_idx[pts_idx_of_class] = class_idx
        
        # identify points with unknown classes --> remove from point cloud
        known_idx = np.where(pts_class_idx != -1)[0]
        self.pcd_filtered = self.pcd_filtered.select_by_index(known_idx)
        print(f"Class of {len(known_idx)} points identified ({len(known_idx) / len(color)} %).")

        return pts_class_idx[known_idx]

    @staticmethod
    def _smoother(pts_idx: np.ndarray, pts_grid: np.ndarray, pts_loss: np.ndarray) -> np.ndarray:
        # get grid idx for each point
        lock.acquire()  # do not access the same memort twice
        pts_loss_local = pts_loss[pts_idx].copy()
        pts_grid_local = pts_grid[pts_idx].copy()
        lock.release()
        
        # parameters
        conv_crit = 0.85
        nb_neigh = 25
        
        # fit kd-tree to available points
        kd_tree = scipy.spatial.KDTree(pts_grid_local)
        pt_dist, pt_neigh_idx = kd_tree.query(pts_grid_local, k=nb_neigh + 1)
        pt_dist = pt_dist[:, 1:]  # filter the point itself
        pt_neigh_idx = pt_neigh_idx[:, 1:]  # filter the point itself
        
        # turn distance into weight
        # pt_dist_weighted = pt_dist * np.linspace(1, 0.01, nb_neigh)     
        pt_dist_inv = 1.0 / pt_dist
        pt_weights = scipy.special.softmax(pt_dist_inv, axis=1)
        
        # smooth losses
        counter = 0
        pts_loss_smooth = pts_loss_local.copy()
        while True:
            counter += 1
            pts_loss_smooth = np.sum(pts_loss_smooth[pt_neigh_idx] * pt_weights, axis=1)
            
            conv_rate = np.sum(np.round(pts_loss_smooth, 3) != np.round(pts_loss_local, 3)) / pts_loss_local.shape[0]            
            if conv_rate > conv_crit:
                print(f"Process {mp.current_process().name} converged with {np.round(conv_rate * 100, decimals=2)} % of changed points after {counter} iterations.")
                break
        
        # lock.acquire()  # do not access the same memort twice
        # smooth_loss[pts_idx] = pts_loss_smooth
        # lock.release()
        
        return pts_loss_smooth
    
    @staticmethod
    def _smoother_init(l: mp.Lock) -> None:
        global lock
        lock = l
        return
    
    def _get_grid_loss(self, class_idx: np.ndarray, class_loss: list) -> np.ndarray:
        """convert points to grid"""
        # get points
        pts = np.asarray(self.pcd_filtered.points)
        pts_grid = (pts[:, :2] - np.array([self._start_x, self._start_y])) / self._cfg_general.resolution
        
        # get loss for each point
        pts_loss = np.zeros(class_idx.shape[0])
        for sem_class in range(len(class_loss)):
            pts_loss[class_idx == sem_class] = class_loss[sem_class]
             
        # split task index
        num_tasks = mp.cpu_count()
        pts_task_idx = np.array_split(np.random.permutation(pts_loss.shape[0]), num_tasks)
        
        # create pool with lock
        l = mp.Lock()
        pool = mp.pool.Pool(processes=num_tasks, initializer=self._smoother_init, initargs=(l,))
        loss_array = pool.map(partial(self._smoother, pts_grid=pts_grid, pts_loss=pts_loss), pts_task_idx)
        pool.close()
        pool.join() 
        
        # reassemble loss array
        smooth_loss = np.zeros_like(pts_loss)
        for process_idx in range(num_tasks):
            smooth_loss[pts_task_idx[process_idx]] = loss_array[process_idx]
        
        if self.visualize:
            plt.scatter(pts[:, 0], pts[:, 1], c=smooth_loss, cmap='jet')
            plt.show()
        
        return smooth_loss
    
    def _dense_grid_loss(self, smooth_loss: np.ndarray) -> None:
        # get grid idx of all classified points
        pts = np.asarray(self.pcd_filtered.points)
        pts_grid_idx = (np.round((pts[:, :2] - np.array([self._start_x, self._start_y])) / self._cfg_general.resolution)).astype(int)
        grid_idx, pts_to_grid_idx_map = np.unique(pts_grid_idx, axis=0, return_index=True)
        
        # assign each point its smoothed loss
        grid_loss = np.ones((self._num_x, self._num_y)) * -1
        grid_loss[grid_idx[:, 0], grid_idx[:, 1]] = smooth_loss[pts_to_grid_idx_map]
        
        # get grid idx of all (non-) classified points
        non_classified_idx = np.where(grid_loss == -1)
        non_classified_idx = np.vstack((non_classified_idx[0], non_classified_idx[1])).T
        
        # fit non-convex hull around points to get grid points outside of mesh
        mesh_shape = alphashape.alphashape(pts_grid_idx)
        inside_mesh = mesh_shape.contains_points(pts_grid_idx[non_classified_idx[:, 0], non_classified_idx[:, 1]])
        non_classified_idx_in_mesh = non_classified_idx[inside_mesh]
        non_classified_idx_out_mesh = non_classified_idx[~inside_mesh]
        
        # fit a k-nearest neighbor regressor to the grid cells with a known loss and regress their loss
        classifier = KNeighborsRegressor(n_neighbors=2, weights="distance", n_jobs=-1)
        classifier.fit(grid_idx, grid_loss[grid_idx])
        grid_loss[non_classified_idx_in_mesh[:, 0], non_classified_idx_in_mesh[:, 1]] = classifier.predict(non_classified_idx_in_mesh)
        
        # assign points outside the mesh the obstacle loss
        grid_loss[non_classified_idx_out_mesh] = OBSTACLE_LOSS
        
        assert any(grid_loss == -1) == False, "There are still grid cells without a loss value."
        
        # plot grid classes and losses
        if self.visualize:
            plt.imshow(grid_loss)
            plt.show()

        return grid_loss
    
    def _smooth_loss(self, grid_loss: np.ndarray) -> np.ndarray:      
        # # apply log
        # grid_loss[grid_loss > 0.0]  = np.log(grid_loss[grid_loss > 0.0] + math.e)
        
        # smooth loss again
        grid_loss = scipy.ndimage.gaussian_filter(grid_loss, sigma=self._cfg_general.sigma_expand*2)        
      
        if self.visualize:
            plt.imshow(grid_loss)
            plt.show()
            
        return grid_loss
    
# EoF
