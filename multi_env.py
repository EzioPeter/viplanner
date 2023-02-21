#!/usr/bin/env python3

# python
import torch
torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from config import TrainCfg, DataCfg
from utils.trainer import Trainer


if __name__ == "__main__":
    # Arguements  
    matterport_overfit: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["2n8kARJN3HM", "2n8kARJN3HM"],
        test_env_id=1,
        file_name="overfit_ratio0.15",
        hierarchical=False,
        data_cfg=DataCfg(
            ratio_fov_samples=0.775,
            ratio_back_samples=0.075,
            ratio_front_samples=0.15,
        )
    )
    trainer = Trainer(matterport_overfit)
    trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()

    matterport_overfit_hierarch: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["2n8kARJN3HM", "2n8kARJN3HM"],
        test_env_id=1,
        file_name="overfit_ratio0.15",
        hierarchical=True,
    )
    trainer = Trainer(matterport_overfit_hierarch)
    trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()
        
    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
    )  
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()
            
    carla: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["town01", "town01"],
        test_env_id=1,
        data_cfg=DataCfg(
            max_goal_distance=10.0,
            max_depth=15,
        ),
        n_visualize=128,
        wb_project="SemNav-Carla"
    )  
    # trainer = Trainer(carla)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache() 

    carla_obscost: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["town01", "town01"],
        test_env_id=1,
        file_name="_obscostheight_05",
        data_cfg=DataCfg(
            max_goal_distance=10.0,
            max_depth=15,
            obs_cost_height=0.5,
            ratio_hard=0.5,
            ratio_easy=0.4,
            ratio_outside=0.1,
        ),
        n_visualize=400,
        wb_project="SemNav-Carla"
    )      
    # trainer = Trainer(carla_obscost)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()

# EoF
