#!/usr/bin/env python3

# python
import torch
torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from config import TrainCfg, DataCfg
from utils.trainer import Trainer


if __name__ == "__main__":
    # Arguements  
    matterport_overfit_hierarch: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["2n8kARJN3HM", "2n8kARJN3HM"],
        test_env_id=1,
        file_name="overfit_ratio09",
        hierarchical=True,
    )
    # trainer = Trainer(matterport_overfit_hierarch)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()
    
    matterport_overfit: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["2n8kARJN3HM", "2n8kARJN3HM"],
        test_env_id=1,
        file_name="overfit_ratio09",
        hierarchical=False,
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
        )
    )
    trainer = Trainer(matterport_overfit)
    # trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.91_back0.03_front0.06",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
        )
    )  
    trainer = Trainer(matterport_sem)
    trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.91_back0.03_front0.06_obsthres1.0",
        obstacle_thred=1.0,
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
        )
    )  
    trainer = Trainer(matterport_sem)
    trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()

    matterport_sem_harder: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.80_back0.07_front0.13",
        data_cfg=DataCfg(
            ratio_fov_samples=0.80,
            ratio_back_samples=0.07,
            ratio_front_samples=0.13,
        )
    )  
    trainer = Trainer(matterport_sem_harder)
    trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()
               
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
            ratio_fov_hard_samples=0.5,
            ratio_fov_easy_samples=0.4,
            ratio_fov_outside_samples=0.1,
        ),
        n_visualize=400,
        wb_project="SemNav-Carla"
    )      
    # trainer = Trainer(carla_obscost)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()

# EoF
