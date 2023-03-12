#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Load Semantics from Matterport3D and make them available to Isaac Sim
"""

# python
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import yaml
import os


# define own loader class to include DataCfg
class Loader(yaml.SafeLoader):
    pass
def construct_datacfg(loader, node):
    add_dicts = {}
    for node_entry in node.value:
        if isinstance(node_entry[1], yaml.MappingNode):
            add_dicts[node_entry[0].value] = loader.construct_mapping(node_entry[1])
            node.value.remove(node_entry)
            
    return DataCfg(**loader.construct_mapping(node), **add_dicts)
Loader.add_constructor('tag:yaml.org,2002:python/object:config.config.DataCfg', construct_datacfg)
# after evaluation in isaac sim, tag changes 
Loader.add_constructor('tag:yaml.org,2002:python/object:omni.isaac.anymal.viplanner.src.config.learning_cfg.DataCfg', construct_datacfg)
        
@dataclass
class DataCfg:
    """Config for data loading (only available for new dataloader --> flag in TrainCfg)"""
    
    # real world data used --> images have to be rotated by 180 degrees
    real_world_data: bool = False 
    
    # identification suffix of the cameras for semantic and depth images
    depth_suffix = "_cam0"
    sem_suffix = "_cam1"
        
    # data processing
    max_depth: float = 15.0
    "maximum depth for depth image"

    # odom (=start) point selection
    max_goal_distance: float = max_depth
    min_goal_distance: float = 0.5
    "maximium and minimum distance between odom and goal"
    distance_scheme: dict = field(default_factory=lambda: {1: 0.2, 3: 0.35, 5: 0.25, 7.5: 0.15, 10: 0.05})
    "select goal points for the samples according to the scheme: {distance: percentage of goals}, distances have to be increasing and max distance has to be equal to max_goal_distance"
    obs_cost_height: float = 0.3
    "all odom points with cost of more than obs_cost_height are discarded"
    fov_scale: float = 1.0
    "scaling of the field of view (only goals within fov are considered)"
    depth_scale: float = 1000.0
    "scaling of the depth image"
    
    # train val split
    ratio: float = 0.9
    "ratio between train and val dataset"
    max_train_pairs: Optional[int] = None
    pairs_per_image: int = 4
    "maximum number of train pairs (can be used to limit training time) can be set, otherwise number of recorded images times pairs_per_image is used"
    ratio_fov_samples: float = 1.0
    ratio_front_samples: float = 0.0
    ratio_back_samples: float = 0.0
    "samples distrubution -> either within the robots fov, in front of the robot but outside the fov or behind the robot"

    # data in memeory
    load_into_memory: bool = True if os.getenv('EXPERIMENT_DIRECTORY') is not None else False
    "load all data into memory (RAM) to speed up training"

@dataclass
class TrainCfg:
    """Config for multi environment training"""
    
    # high level configurations
    sem: bool = True 
    "use semantic image"
    file_name: Optional[str] = None
    "appendix to the filename if needed"      
    seed: int = 0
    "random seed"  
    gpu_id: int = 0 
    "GPU id"      
    
    # data and dataloader configurations
    cost_map_name: str = "cost_map_sem" # "cost_map_sem" 
    "cost map name"
    env_list: List[str] = field(default_factory=lambda: 
        ["2azQ1b91cZZ",
         "JeFG25nYj2p",
         "Vvot9Ly1tCj",
         "ur6pFq6Qu1A",
         "B6ByNegPMKs",
         "8WUmhLawc2A",
         "E9uDoFAP3SH",
         "QUCTc6BB5sX",
         "YFuZgdQ5vWj",
         "2n8kARJN3HM"]
    )
    test_env_id: int = 5
    "the test env id in the id list"    
    data_cfg: DataCfg = DataCfg()
    "further data configuration"
    multi_epoch_dataloader: bool = True
    "load all samples into RAM s.t. do not have to be reloaded for each epoch"   
    num_workers: int = 4
    "number of workers for dataloader"     
    
    # loss configurations
    fear_ahead_dist: float =2.5 
    "fear lookahead distance"
    w_obs: float = 0.25
    w_height: float = 1.0
    w_motion: float = 1.5
    w_goal: float = 2.0
    "weights for the loss components"
    obstacle_thred: float = 0.75
    "obstacle threshold to decide if fear path or not"  
    
    # network configurations
    img_input_size: Tuple[int, int] = field(default_factory=lambda: [360, 640]) 
    "image size (will be cropped if larger or resized if smaller)"    
    in_channel: int = 16 
    "goal input channel numbers"
    knodes: int = 5 
    "number of max waypoints predicted"    

    # training configurations
    resume: bool = False
    "resume training"    
    epochs: int = 100
    "number of training epochs"    
    batch_size: int = 64 
    "number of minibatch size"    
    hierarchical: bool = False
    hierarchical_step: int = 50
    hierarchical_front_step_ratio: float = 0.02
    hierarchical_back_step_ratio: float = 0.01
    "hierarchical training with an adjusted data structure"
    
    # optimizer and scheduler configurations
    lr: float = 2e-3 
    "learning rate"
    factor: float = 0.5 
    "ReduceLROnPlateau factor"
    min_lr: float = 1e-5
    "minimum lr for ReduceLROnPlateau"
    patience: int = 3 
    "patience of epochs for ReduceLROnPlateau"    
    optimizer: str = "sgd"  # either adam or sgd
    "optimizer"
    momentum: float = 0.1 
    "momentum of the optimizer"
    w_decay: float = 1e-4 
    "weight decay of the optimizer"
    
    # visualization configurations
    camera_tilt: float = 0.15 
    "camera tilt angle for visualization only"
    n_visualize: int = 15
    "number of trajectories that are visualized"

    # logging configurations
    wb_project: str = "SemNav-Matterport"
    wb_entity: str = "semnav"
    wb_api_key: str = "e718d064556efc09b0bd0574a8e458f92dea49fc"
    
    # functions
    def _get_model_save(self, epoch: Optional[int] = None):
        input_domain = "DepSem" if self.sem else "Dep"
        cost_name = "Geom" if self.cost_map_name == "cost_map_geom" else "Sem"
        optim = "SGD" if self.optimizer == "sgd" else "Adam"
        name = f"_{self.file_name}" if self.file_name is not None else ""
        epoch = epoch if epoch is not None else self.epochs
        hierarch = f"_hierarch" if self.hierarchical else ""
        return f"plannernet_env{self.env_list[0]}_ep{epoch}_input{input_domain}_cost{cost_name}_optim{optim}{hierarch}{name}"

    @classmethod
    def from_yaml(cls, yaml_path: str):
        # open yaml file and load config
        with open(yaml_path, "r") as f:
            cfg_dict = yaml.load(f, Loader=Loader)
        
        return cls(**cfg_dict["config"])
# EoF
