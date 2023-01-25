#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Load Semantics from Matterport3D and make them available to Isaac Sim
"""

# python
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class DataCfg:
    """Config for data loading (only available for new dataloader --> flag in TrainCfg)"""
    
    # real world data used --> images have to be rotated by 180 degrees
    real_world_data: bool = False 

    # data processing
    max_depth: float = 15.0
    "maximum depth for depth image"
    max_goal_per_odom: int = 5
    "maximum number of goals per odom (=start) point"

    # odom (=start) point selection
    max_goal_distance: float = 10.0
    "maximum distance between odom and goal"
    min_goal_distance: float = 1.0
    "minimum distance between odom and goal"
    n_rays_check: int = 15
    ray_obs_ratio: float = 0.8
    "number of rays to check for obstacles between odom and goal -> if over ray_obs_ratio, odom is discarded"
    obs_cost_height: float = 0.01
    "all odom points with cost of more than obs_cost_height are discarded"
    free_space_cost_height: float = 0.01
    """odom points after all filtering with cost heigher can be weighted in the neural network cost"""
    fov_scale: float = 1.0
    "scaling of the field of view (only goals within fov are considered)"
    
    # train val split
    ratio: float = 0.9
    "ratio between train and val dataset"
    max_train_pairs: int = 10000  # difficult samples for sure included
    "maximum number of train pairs (can be used to limit training time)"
    ratio_hard: float = 0.5
    ratio_easy: float = 0.5
    ratio_outside: float = 0.0


@dataclass
class TrainCfg:
    """Config for multi environment training"""
    
    data_root: str = "data_pc" 
    "dataset root folder"
    sem: bool = True 
    "use semantic image"
    cost_map_name: str = "cost_map_geom" # "cost_map_sem" 
    "cost map name"
    camera_tilt: float = 0.15 
    "camera tilt angle for visualization only"
    sensor_offsetX_ANYmal: float = 0.4 
    "anymal front camera sensor offset in X axis"
    env_list: List[str] = field(default_factory=lambda: ["2azQ1b91cZZ", "JeFG25nYj2p", "Vvot9Ly1tCj", "ur6pFq6Qu1A", "B6ByNegPMKs", "2n8kARJN3HM"])
    test_env_id: int = 5
    "the test env id in the id list"
    resume: bool = False
    crop_size: Tuple[int, int] = field(default_factory=lambda: [360, 640]) 
    "image crop size"
    in_channel: int = 16 
    "goal input channel numbers"
    knodes: int = 5 
    "number of max waypoints predicted"
    lr: float = 2e-3 
    "learning rate"
    factor: float = 0.5 
    "ReduceLROnPlateau factor"
    min_lr: float = 1e-6 
    "minimum lr for ReduceLROnPlateau"
    patience: int = 10 
    "patience of epochs for ReduceLROnPlateau"
    epochs: int = 200
    "number of training epochs"
    batch_size: int = 64 
    "number of minibatch size"
    optimizer: str = "sgd"  # either adam or sgd
    "optimizer"
    momentum: float = 0.1 
    "momentum of the optimizer"
    w_decay: float = 1e-4 
    "weight decay of the optimizer"
    num_workers: int = 6 
    "number of workers for dataloader"
    gpu_id: int = 0 
    "GPU id"
    training: bool = False
    "the dataset type"
    fear_ahead_dist: float =2.5 
    "fear lookahead distance"
    old_dataloader: bool = False
    "use old dataloader for testing reasons -> only with depth information and old structure"
    seed: int = 0
    "random seed"
    file_name: Optional[str] = None
    "appendix to the filename if needed"
    n_visualize: int = 10
    "number of trajectories that are visualized"
    weight_samples_difficult: float = 1.0  # start-goal combinations with obstacle inbetween (cost value higher than data_cfg.obs_cost_height)
    weight_samples_high_cost: float = 1.0  # start-goal combinations with either start or end point with higher cost than data_cfg.free_space_cost_height
    """weighting of samples with special conditions"""
    
    # logging
    wb_project: str = "SemNav-Matterport"
    wb_entity: str = "semnav"
    wb_api_key: str = "e718d064556efc09b0bd0574a8e458f92dea49fc"
    
    # further config class
    data_cfg: DataCfg = DataCfg()
    
    # functions
    def _get_model_save(self):
        input_domain = "DepSem" if self.sem else "Dep"
        cost_name = "Geom" if self.cost_map_name == "cost_map_geom" else "Sem"
        optim = "SGD" if self.optimizer == "sgd" else "Adam"
        name = f"_{self.file_name}" if self.file_name is not None else ""
        return f"plannernet_{self.env_list[0]}_ep{self.epochs}_input{input_domain}_cost{cost_name}_optim{optim}{name}.pt"
# EoF
    