#!/usr/bin/env python3

# python
import torch
torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from viplanner.config import TrainCfg, DataCfg
from viplanner.utils.trainer import Trainer


if __name__ == "__main__":
    """ ANYMAL-C Training """
    env_list_anymal_c = [
        "2azQ1b91cZZ_anymal_c",
        "JeFG25nYj2p_anymal_c",
        "Vvot9Ly1tCj_anymal_c",
        "ur6pFq6Qu1A_anymal_c",
        "B6ByNegPMKs_anymal_c",
        "8WUmhLawc2A_anymal_c",
        "E9uDoFAP3SH_anymal_c",
        "QUCTc6BB5sX_anymal_c",
        "YFuZgdQ5vWj_anymal_c",
        "2n8kARJN3HM_anymal_c"
    ]
    
    matterport_anymal_c: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="anymal_c",
        data_cfg=DataCfg(
            ratio_fov_samples=1.0,
            ratio_back_samples=0.0,
            ratio_front_samples=0.0,
        ),
        env_list=env_list_anymal_c,
    )  
    # trainer = Trainer(matterport_anymal_c)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_anymal_c_neg05: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        file_name="anymal_c_neg05",
        data_cfg=DataCfg(
            ratio_fov_samples=1.0,
            ratio_back_samples=0.0,
            ratio_front_samples=0.0,
        ),
        env_list=env_list_anymal_c,
    )  
    # trainer = Trainer(matterport_anymal_c_neg05)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_anymal_c_neg10: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg10",
        file_name="anymal_c_neg10",
        data_cfg=DataCfg(
            ratio_fov_samples=1.0,
            ratio_back_samples=0.0,
            ratio_front_samples=0.0,
        ),
        env_list=env_list_anymal_c,
    )  
    # trainer = Trainer(matterport_anymal_c_neg10)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()
        
    matterport_anymal_c: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="anymal_c",
        data_cfg=DataCfg(
            ratio_fov_samples=1.0,
            ratio_back_samples=0.0,
            ratio_front_samples=0.0,
        ),
        env_list=env_list_anymal_c,
    )  
    # trainer = Trainer(matterport_anymal_c)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()
    
    matterport_anymal_c: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="anymal_c_fov0.91_back0.03_front0.06",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
        ),
        env_list=env_list_anymal_c,
    )  
    # trainer = Trainer(matterport_anymal_c)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_anymal_c_noise: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        file_name="anymal_c_fov0.91_back0.03_front0.06_all_noise",
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
        ),
        env_list=env_list_anymal_c,
    )  
    # trainer = Trainer(matterport_anymal_c_noise)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    ### CARLA 
    
    carla_anymal_c: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        env_list=["town01_anymal_c_new", "town01_anymal_c_new"],
        test_env_id=1,
        file_name="neg05",
        n_visualize=128,
        wb_project="SemNav-Carla",
    )  
    # trainer = Trainer(carla_anymal_c)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache() 

    carla_anymal_c_out_fov: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["town01_anymal_c", "town01_anymal_c"],
        test_env_id=1,
        n_visualize=128,
        file_name="fov0.91_back0.03_front0.06",
        wb_project="SemNav-Carla",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
        ),
    )  
    # trainer = Trainer(carla_anymal_c_out_fov)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache() 
        
    carla_anymal_c_neg10: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg10",
        env_list=["town01_anymal_c_new", "town01_anymal_c_new"],
        test_env_id=1,
        file_name="neg10",
        n_visualize=128,
        wb_project="SemNav-Carla",

    )  
    # trainer = Trainer(carla_anymal_c_neg10)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    carla_anymal_c_out_fov_neg05: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        env_list=["town01_anymal_c", "town01_anymal_c"],
        test_env_id=1,
        n_visualize=128,
        file_name="fov0.91_back0.03_front0.06_neg05",
        wb_project="SemNav-Carla",
        data_cfg=DataCfg(
            ratio_fov_samples=0.91,
            ratio_back_samples=0.03,
            ratio_front_samples=0.06,
        ),
    )  
    # trainer = Trainer(carla_anymal_c_out_fov_neg05)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache() 
    
    """ COMBINED TRAINING """
    env_list_combi = [
        "2azQ1b91cZZ_anymal_c",
        "JeFG25nYj2p_anymal_c",
        "Vvot9Ly1tCj_anymal_c",
        "town01_anymal_c_new",
        "ur6pFq6Qu1A_anymal_c",
        "B6ByNegPMKs_anymal_c",        
        "8WUmhLawc2A_anymal_c",
        "town01_anymal_c_new",
    ]

    matterport_anymal_c_neg05_combi: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        file_name="anymal_c_combi",
        data_cfg=DataCfg(
            ratio_fov_samples=1.0,
            ratio_back_samples=0.0,
            ratio_front_samples=0.0,
        ),
        env_list=env_list_combi,
    )  
    
    # trainer = Trainer(matterport_anymal_c_neg05_combi)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()  