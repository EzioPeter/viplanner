#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      config class for reconstruction and cost maps
"""

# python
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class ReconstructionCfg:
    """
    Arguments for 3D reconstruction using depth maps
    """
    # directory where the environment with the depth (and semantic) images is located
    data_dir: str = "/home/pascal/SemNav/env/matterport/data_domains"
    # directory where the reconstructed 3D map is saved
    out_dir: str = "/home/pascal/SemNav/env/matterport/data_pc"
    # environment name
    env: str = "2n8kARJN3HM"

    # reconstruction parameters
    voxel_size: float = 0.04
    start_idx: int = 0  # start index for reconstruction
    max_images: Optional[int] = None  # maximum number of images to reconstruct, fi None, all images are used

    # semantic reconstruction
    semantics: bool = True

    """ Internal functions """    
    def get_data_path(self) -> str:
        return os.path.join(self.data_dir, self.env)
    
    def get_out_path(self) -> str:
        return os.path.join(self.out_dir, self.env)


@dataclass
class SemCostMapConfig:
    """Configuration for the semantic cost map"""
    # path to point cloud
    ply_path: str = "/home/pascal/SemNav/env/matterport/data_pc/2n8kARJN3HM/cloud.ply"
    # resolution of the cost map
    resolution: float = 0.01
    # filter parameters
    ground_height: float = 0.0
    robot_height: float = 0.70
    robot_height_factor: float = 1.5
    nb_neighbors: int = 100
    std_ratio: float = 1.0
    # map parameters
    clear_dist: float = 1.0  # cost map expansion over the point cloud space (prevent paths to go out of the map)
    # smooting parameters
    sigma_expand: float = 1.0
    sigma_smooth: float = 2.0
    # color mapping
    data_source: str = "matterport"  # "matterport" or "carla"
    mapping_dir: str = "/home/pascal/SemNav/env/matterport/data_domains/2n8kARJN3HM/mapping"
    # obstacle dilation
    dilation: int = 15  # dilation of obstacles in grid cells (grid cell size = resolution)
    

# EoF
