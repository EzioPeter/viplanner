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
    # filter parameters
    ground_height: float = -0.05
    robot_height: float = 0.70
    robot_height_factor: float = 1.5
    nb_neighbors: int = 100
    std_ratio: float = 2.0  # keep high, otherwise ground will be removed
    # color mapping
    data_source: str = "matterport"  # "matterport" or "carla"
    mapping_dir: str = "/home/pascal/SemNav/env/matterport/data_domains/2n8kARJN3HM/mapping"
    

@dataclass
class TsdfCostMapConfig:
    """Configuration for the tsdf cost map"""
    # offset of the point cloud 
    offset_z: float = 0.0
    # filter parameters
    ground_height: float = 0.25
    robot_height: float = 0.70
    robot_height_factor: float = 1.5
    nb_neighbors: int = 50
    std_ratio: float = 0.5
    filter_outliers: bool = True


@ dataclass 
class GeneralCostMapConfig:
    """General Cost Map Configuration"""
    # path to point cloud
    root_path: str = "/home/pascal/SemNav/env/matterport/data_pc/2n8kARJN3HM"
    ply_file: str = "cloud.ply"
    # resolution of the cost map
    resolution: float = 0.01 
    # map parameters
    clear_dist: float = 1.0  # cost map expansion over the point cloud space (prevent paths to go out of the map)
    # smoothing parameters
    sigma_smooth: float = 2.0
    # dilation parameters
    sigma_expand: float = 1.0
    sigma_expand_free: float = 0.5
    obstacle_threshold: float = 0.01
    free_space_threshold: float = 0.4

@dataclass
class CostMapConfig:
    """General Cost Map Configuration"""
    # cost map domains
    semantics: bool = True
    geometry: bool = True
    
    # name
    map_name: str = "tsdf_sem_1"
    
    # general cost map configuration
    general: GeneralCostMapConfig = GeneralCostMapConfig()
    
    # individual cost map configurations
    sem_cost_map: SemCostMapConfig = SemCostMapConfig()
    tsdf_cost_map: TsdfCostMapConfig = TsdfCostMapConfig()
    
    # visualize cost map
    visualize: bool = False
# EoF
