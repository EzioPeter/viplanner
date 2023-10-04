# python
import torch

torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from viplanner.config import DataCfg, TrainCfg
from viplanner.utils.trainer import Trainer

if __name__ == "__main__":
    test: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_sharpend",
        env_list=["JeFG25nYj2p_cam_mount", "JeFG25nYj2p_cam_mount"],
        test_env_id=1,
        file_name="overfit_test",
        data_cfg=DataCfg(
            carla=True,
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
    # trainer = Trainer(test)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    torch.cuda.empty_cache()

    """ INDOOR TRAINING (AS TEST)"""
    env_list_indoor = [
        "2azQ1b91cZZ_cam_mount",
        "JeFG25nYj2p_cam_mount",
        "warehouse_multiple_shelves_without_ppl",
        "Vvot9Ly1tCj_cam_mount",
        "E9uDoFAP3SH_cam_mount",
        "ur6pFq6Qu1A_cam_mount",
        "warehouse_multiple_shelves_without_ppl",
        "B6ByNegPMKs_cam_mount",
        "8WUmhLawc2A_cam_mount",
        "QUCTc6BB5sX_cam_mount",
        "warehouse_multiple_shelves_without_ppl",
        "YFuZgdQ5vWj_cam_mount",
        "2n8kARJN3HM_cam_mount",
    ]
    indoor: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_sharpend",
        env_list=env_list_indoor,
        test_env_id=12,
        file_name="new_colorspace_sharpend_indoor_lossWidthMod_wgoal4.0_warehouse",
        data_cfg=DataCfg(
            ratio_fov_samples=0.96,
            ratio_back_samples=0.01,
            ratio_front_samples=0.03,
        ),
        wb_project="SemNav-NewColorspace",
    )
    # trainer = Trainer(indoor)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    indoor_noise: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_sharpend",
        env_list=env_list_indoor,
        test_env_id=12,
        file_name="new_colorspace_sharpend_indoor_noise_lossWidthMod_wgoal4.0_warehouse",
        data_cfg=DataCfg(
            ratio_fov_samples=0.96,
            ratio_back_samples=0.01,
            ratio_front_samples=0.03,
            # add noise
            depth_gaussian=None,
            sem_rgb_black_img=0.05,
            noise_edges=True,
            sem_rgb_random_polygons_nb=20,
            depth_random_polygons_nb=20,
        ),
        wb_project="SemNav-NewColorspace",
    )
    # trainer = Trainer(indoor_noise)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

    """ COMBINED TRAINING """
    env_list_combi = [
        "2azQ1b91cZZ_cam_mount",
        "JeFG25nYj2p_cam_mount",
        # "warehouse_multiple_shelves_without_ppl",
        "town01_cam_mount_train",
        "Vvot9Ly1tCj_cam_mount",
        "E9uDoFAP3SH_cam_mount",
        # "warehouse_multiple_shelves_without_ppl",
        "ur6pFq6Qu1A_cam_mount",
        "B6ByNegPMKs_cam_mount",
        # "warehouse_multiple_shelves_without_ppl",
        "8WUmhLawc2A_cam_mount",
        "town01_cam_mount_train",
        "QUCTc6BB5sX_cam_mount",
        # "warehouse_multiple_shelves_without_ppl",
        "YFuZgdQ5vWj_cam_mount",
        "2n8kARJN3HM_cam_mount",
    ]
    samples_cls_parameters = {
        "ratio_fov_samples": 0.96,
        "ratio_back_samples": 0.01,
        "ratio_front_samples": 0.03,
    }
    data_cfg_list = [
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        # DataCfg(**samples_cls_parameters),
        DataCfg(
            carla=True, distance_scheme={4: 0.05, 7.5: 0.60, 10: 0.30}, max_train_pairs=3000, **samples_cls_parameters
        ),
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        # DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        # DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        DataCfg(
            carla=True, distance_scheme={4: 0.05, 7.5: 0.60, 10: 0.30}, max_train_pairs=3000, **samples_cls_parameters
        ),
        DataCfg(**samples_cls_parameters),
        # DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
    ]
    combi: TrainCfg = TrainCfg(
        sem=False,
        cost_map_name="tsdf_sem",
        env_list=env_list_combi,
        test_env_id=12,
        file_name="new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse_depth",
        data_cfg=data_cfg_list,
        n_visualize=16,
        wb_project="SemNav-NewColorspace",
    )
    trainer = Trainer(combi)
    trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()

    noise_parameters = {
        # add nois
        "depth_gaussian": None,
        "sem_rgb_black_img": 0.05,
        "noise_edges": True,
        "sem_rgb_random_polygons_nb": 20,
        "depth_random_polygons_nb": 20,
    }

    data_cfg_list = [
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(
            carla=True,
            distance_scheme={5: 0.20, 7.5: 0.45, 10: 0.30},
            max_train_pairs=3000,
            **samples_cls_parameters,
            **noise_parameters,
        ),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(
            carla=True,
            distance_scheme={5: 0.20, 7.5: 0.45, 10: 0.30},
            max_train_pairs=3000,
            **samples_cls_parameters,
            **noise_parameters,
        ),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
    ]

    combi_noise: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_sharpend",
        env_list=env_list_combi,
        test_env_id=11,
        file_name="new_colorspace_sharpend_combi_noise_lossWidthMod_wgoal4.0_warehouse",
        data_cfg=data_cfg_list,
        n_visualize=16,
        wb_project="SemNav-NewColorspace",
    )
    # trainer = Trainer(combi_noise)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()
