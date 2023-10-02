# python
import torch

torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from viplanner.config import DataCfg, TrainCfg
from viplanner.utils.trainer import Trainer

if __name__ == "__main__":
    # Arguments
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
        ),
    )
    # trainer = Trainer(matterport_depth_noise)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_overfit: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["town01", "town01"],
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
        ),
    )
    # trainer = Trainer(matterport_overfit)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        file_name="neg05",
        data_cfg=DataCfg(
            ratio_fov_samples=1.0,
            ratio_back_samples=0.0,
            ratio_front_samples=0.0,
        ),
    )
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

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
        ),
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
        ),
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
        ),
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
        ),
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
        ),
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
        ),
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
        ),
    )
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        file_name="neg05_fov0.91_back0.03_front0.06_all_noise",
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
    )
    # trainer = Trainer(matterport_sem)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        file_name="new_loss_neg05",
        data_cfg=DataCfg(
            ratio_fov_samples=0.96,
            ratio_back_samples=0.01,
            ratio_front_samples=0.03,
        ),
    )
    trainer = Trainer(matterport_sem)
    # trainer.train()
    trainer.test()
    # trainer.save_config()
    torch.cuda.empty_cache()

    matterport_sem_harder: TrainCfg = TrainCfg(
        sem=False,
        rgb=True,
        cost_map_name="cost_map_sem",
        file_name="fov0.80_back0.07_front0.13_resnet50",
        data_cfg=DataCfg(
            ratio_fov_samples=0.80,
            ratio_back_samples=0.07,
            ratio_front_samples=0.13,
        ),
    )
    # trainer = Trainer(matterport_sem_harder)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    carla: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        env_list=["town01_more_data_train", "town01_more_data_train"],
        test_env_id=1,
        file_name="more_data_neg05",
        data_cfg=DataCfg(
            max_goal_distance=10.0,
        ),
        n_visualize=128,
        wb_project="SemNav-Carla",
    )
    # trainer = Trainer(carla)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    """ COMBINED TRAINING """
    env_list_combi = [
        "2azQ1b91cZZ",
        "JeFG25nYj2p",
        "town01_more_data_train",
        "Vvot9Ly1tCj",
        "E9uDoFAP3SH",
        "ur6pFq6Qu1A",
        "B6ByNegPMKs",
        "8WUmhLawc2A",
        "town01_more_data_train",
        "QUCTc6BB5sX",
        "YFuZgdQ5vWj",
        "2n8kARJN3HM",
    ]
    data_cfg_list = [
        DataCfg(),
        DataCfg(),
        DataCfg(
            carla=True,
            distance_scheme={5: 0.10, 7.5: 0.40, 10: 0.30, 15: 0.15},
            max_train_pairs=15000,
        ),
        DataCfg(),
        DataCfg(),
        DataCfg(),
        DataCfg(),
        DataCfg(),
        DataCfg(
            carla=True,
            distance_scheme={5: 0.10, 7.5: 0.40, 10: 0.30, 15: 0.15},
            max_train_pairs=15000,
        ),
        DataCfg(),
        DataCfg(),
        DataCfg(),
    ]
    combi: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        env_list=env_list_combi,
        test_env_id=11,
        file_name="combi_more_data_neg05",
        data_cfg=data_cfg_list,
        n_visualize=16,
        wb_project="SemNav-Carla",
    )
    trainer = Trainer(combi)
    # trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()

    """ RGB TRAINING """

    env_list_rgb = [
        "2azQ1b91cZZ",
        "JeFG25nYj2p",
        "Vvot9Ly1tCj",
        "ur6pFq6Qu1A",
        "B6ByNegPMKs",
        "8WUmhLawc2A",
        "E9uDoFAP3SH",
        "QUCTc6BB5sX",
        "YFuZgdQ5vWj",
        "2n8kARJN3HM",
    ]

    matterport_rgb: TrainCfg = TrainCfg(
        sem=False,
        rgb=True,
        cost_map_name="cost_map_sem_neg05",
        file_name="rgb",
        data_cfg=DataCfg(
            ratio_fov_samples=1.0,
            ratio_back_samples=0.0,
            ratio_front_samples=0.0,
        ),
        env_list=env_list_rgb,
    )
    # trainer = Trainer(matterport_rgb)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()


# EoF
