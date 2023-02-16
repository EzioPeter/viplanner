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
    data_dir: str = "/home/pascal/SemNav/imperative_learning/data"
    # environment name
    env: str = "2n8kARJN3HM"  # Vvot9Ly1tCj ur6pFq6Qu1A B6ByNegPMKs 2azQ1b91cZZ  town01 town01_2 JeFG25nYj2p 
    # image suffix
    depth_suffix = "_cam0"
    sem_suffix = "_cam1"
    # higher resolution depth images available for reconstruction
    high_res_depth: bool = False
    res_factor: int = 2  # resolution factor for high resolution depth images
    
    # reconstruction parameters
    voxel_size: float = 0.05  # [m]
    start_idx: int = 0  # start index for reconstruction
    max_images: Optional[int] = 300  # maximum number of images to reconstruct, if None, all images are used
    depth_scale: float = 1000.0  # depth scale factor
    # semantic reconstruction
    semantics: bool = True

    # speed vs. memory trade-off parameters
    point_cloud_batch_size: int = 200  # 3d points of nbr images added to point cloud at once (higher values use more memory but faster)

    """ Internal functions """    
    def get_data_path(self) -> str:
        return os.path.join(self.data_dir, self.env)
    
    def get_out_path(self) -> str:
        return os.path.join(self.out_dir, self.env)


@dataclass
class SemCostMapConfig:
    """Configuration for the semantic cost map"""
    # point-cloud filter parameters
    ground_height: Optional[float] = None  # -0.05 for matterport
    robot_height: float = 0.70
    robot_height_factor: float = 2.0
    nb_neighbors: int = 100
    std_ratio: float = 2.0  # keep high, otherwise ground will be removed
    downsample: bool = False
    # color mapping
    data_source: str = "matterport"  # "matterport" or "carla"
    mapping_dir: str = "/home/pascal/SemNav/orbit/source/extensions/omni.isaac.matterport/data/mappings"  # only needed for matterport
    # smooting
    nb_neigh: int = 15
    change_decimal: int = 3
    conv_crit: float = 0.45  # ration of points that have to change by at least the #change_decimal decimal value to converge  
    nb_tasks: Optional[int] = 10  # number of tasks for parallel processing, if None, all available cores are used
    sigma_smooth: float = 2.0
    max_iterations: int = 1
    # obstacle threshold
    obstacle_threshold: float = 0.3
    # loss values rounded up to decimal #round_decimal_traversable equal to 0.0 are selected and the traversable gradient is determined based on them
    round_decimal_traversable: int = 2


@dataclass
class TsdfCostMapConfig:
    """Configuration for the tsdf cost map"""
    # offset of the point cloud 
    offset_z: float = 0.0
    # filter parameters
    ground_height: float = 0.35
    robot_height: float = 0.70
    robot_height_factor: float = 2.0
    nb_neighbors: int = 50
    std_ratio: float = 0.2
    filter_outliers: bool = True
    # dilation parameters
    sigma_expand: float = 2.0
    obstacle_threshold: float = 0.01
    free_space_threshold: float = 0.5


@ dataclass 
class GeneralCostMapConfig:
    """General Cost Map Configuration"""
    # path to point cloud
    root_path: str = "/home/pascal/SemNav/imperative_learning/data/2n8kARJN3HM"  # B6ByNegPMKs JeFG25nYj2p Vvot9Ly1tCj ur6pFq6Qu1A  town01 2azQ1b91cZZ 
    ply_file: str = "cloud.ply"
    # resolution of the cost map
    resolution: float = 0.04  # [m]  (0.04 for matterport, 0.1 for carla)
    # map parameters
    clear_dist: float = 1.0  # cost map expansion over the point cloud space (prevent paths to go out of the map)
    # smoothing parameters
    sigma_smooth: float = 2.0
    # cost map expansion
    x_min: Optional[float] = None  # -8.05  # [m] if None, the minimum of the point cloud is used
    y_min: Optional[float] = None  # -8.05  # [m] if None, the minimum of the point cloud is used
    x_max: Optional[float] = None  # 402.38 # [m] if None, the maximum of the point cloud is used
    y_max: Optional[float] = None  # 336.65 # [m] if None, the maximum of the point cloud is used


@dataclass
class CostMapConfig:
    """General Cost Map Configuration"""
    # cost map domains
    semantics: bool = True
    geometry: bool = False
    
    # name
    map_name: str = "cost_map_sem"
    
    # general cost map configuration
    general: GeneralCostMapConfig = GeneralCostMapConfig()
    
    # individual cost map configurations
    sem_cost_map: SemCostMapConfig = SemCostMapConfig()
    tsdf_cost_map: TsdfCostMapConfig = TsdfCostMapConfig()
    
    # visualize cost map
    visualize: bool = True
# EoF
