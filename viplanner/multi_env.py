#!/usr/bin/env python3

# python
import torch
torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from viplanner.config import TrainCfg, DataCfg
from viplanner.utils.trainer import Trainer


if __name__ == "__main__":
    # Arguements  
    matterport_depth: TrainCfg = TrainCfg(
        sem=False,
        rgb=False,
        cost_map_name="cost_map_sem",
        file_name="depth",
    )
    # trainer = Trainer(matterport_depth)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_depth_outside: TrainCfg = TrainCfg(
        sem=False,
        rgb=False,
        cost_map_name="cost_map_sem",
        file_name="depth_fov0.91_back0.03_front0.06",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
        )
    )
    # trainer = Trainer(matterport_depth_outside)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_depth_noise: TrainCfg = TrainCfg(
        sem=False,
        rgb=False,
        cost_map_name="cost_map_sem",
        file_name="depth_fov0.91_back0.03_front0.06_noise",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            depth_random_polygons_nb=10,
            sem_rgb_random_polygons_nb=10,
            noise_edges=True,
            depth_salt_pepper=0.05,
            depth_gaussian=0.05,
            sem_rgb_black_img=0.05,
        )
    )
    # trainer = Trainer(matterport_depth_noise)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_overfit_hierarch: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["JeFG25nYj2p", "JeFG25nYj2p"],
        test_env_id=1,
        file_name="test",
        hierarchical=False,
    )
    trainer = Trainer(matterport_overfit_hierarch)
    trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()
    
    matterport_overfit: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["2n8kARJN3HM", "2n8kARJN3HM"],
        test_env_id=1,
        file_name="overfit_test",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            depth_gaussian=0.05,
            sem_rgb_black_img=0.05,
            noise_edges=True,
            sem_rgb_random_polygons_nb=10,
            depth_random_polygons_nb=10,
        )
    )
    # trainer = Trainer(matterport_overfit)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_obs04",
        file_name="obs04_sdecoder",
        data_cfg=DataCfg(
            ratio_fov_samples=1.0,
            ratio_back_samples=0.0,
            ratio_front_samples=0.0,
        )
    )  
    trainer = Trainer(matterport_sem)
    trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_obs04",
        file_name="obs04_fov0.91_back0.03_front0.06_sdecoder",
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
        file_name="fov0.91_back0.03_front0.06_sdecoder_noise_depthSP",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            depth_salt_pepper=0.05,
            depth_gaussian=None,
        )
    )  
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.91_back0.03_front0.06_sdecoder_noise_depthGauss",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            depth_salt_pepper=None,
            depth_gaussian=0.05,
        )
    )  
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.91_back0.03_front0.06_sdecoder_noise_depth",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            depth_salt_pepper=0.05,
            depth_gaussian=0.05,
        )
    )  
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.91_back0.03_front0.06_sdecoder_sem_black",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            sem_rgb_black_img=0.05,
        )
    )
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.91_back0.03_front0.06_sdecoder_noise_depth_sem_black",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            depth_salt_pepper=0.05,
            depth_gaussian=0.05,
            sem_rgb_black_img=0.05,
        )
    )  
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.91_back0.03_front0.06_sdecoder_edge_noise",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            noise_edges=True,
        )
    )  
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()
 
    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.91_back0.03_front0.06_sdecoder_polygons",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            depth_random_polygons_nb=10,
            sem_rgb_random_polygons_nb=10,
        )
    )  
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_obs04",
        file_name="obs04_fov0.91_back0.03_front0.06_sdecoder_all_noise",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
            depth_random_polygons_nb=10,
            sem_rgb_random_polygons_nb=10,
            noise_edges=True,
            depth_salt_pepper=0.05,
            depth_gaussian=0.05,
            sem_rgb_black_img=0.05,
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
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem_harder: TrainCfg = TrainCfg(
        sem=False,
        rgb=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.80_back0.07_front0.13_resnet50",
        data_cfg=DataCfg(
            ratio_fov_samples=0.80,
            ratio_back_samples=0.07,
            ratio_front_samples=0.13,
        )
    )  
    # trainer = Trainer(matterport_sem_harder)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()
               
    carla: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["town01", "town01"],
        test_env_id=1,
        file_name="resnet50",
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
        ),
        n_visualize=400,
        wb_project="SemNav-Carla"
    )      
    # trainer = Trainer(carla_obscost)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()

    carla_fargoal20: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["town01", "town01"],
        test_env_id=1,
        file_name="fargoal20",
        data_cfg=DataCfg(
            max_goal_distance=20.0,
            max_depth=15,
            distance_scheme={5: 0.2, 10: 0.35, 15: 0.25, 17.5: 0.10, 20: 0.10},
            ratio_fov_samples=0.90,
            ratio_back_samples=0.03,
            ratio_front_samples=0.07,
        ),
        n_visualize=128,
        wb_project="SemNav-Carla"
    )  
    # trainer = Trainer(carla_fargoal20)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache() 

    carla_fargoal30: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["town01", "town01"],
        test_env_id=1,
        file_name="fargoal30",
        data_cfg=DataCfg(
            max_goal_distance=30.0,
            max_depth=15,
            distance_scheme={5: 0.2, 10: 0.35, 15: 0.25, 20: 0.10, 30: 0.10},
            ratio_fov_samples=0.90,
            ratio_back_samples=0.03,
            ratio_front_samples=0.07,
        ),
        n_visualize=128,
        wb_project="SemNav-Carla"
    )  
    # trainer = Trainer(carla_fargoal30)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache() 
    
# EoF
