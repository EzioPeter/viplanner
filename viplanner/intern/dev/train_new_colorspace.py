# python
import torch

torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from viplanner.config import DataCfg, TrainCfg
from viplanner.utils.trainer import Trainer

if __name__ == "__main__":
    """INDOOR TRAINING (AS TEST)"""
    env_list_indoor = [
        "2azQ1b91cZZ_new_colorspace",
        "JeFG25nYj2p_new_colorspace",
        "Vvot9Ly1tCj_new_colorspace",
        "E9uDoFAP3SH_new_colorspace",
        "ur6pFq6Qu1A_new_colorspace",
        "B6ByNegPMKs_new_colorspace",
        "8WUmhLawc2A_new_colorspace",
        "QUCTc6BB5sX_new_colorspace",
        "YFuZgdQ5vWj_new_colorspace",
        "2n8kARJN3HM_new_colorspace",
    ]
    indoor: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_sharpend",
        env_list=env_list_indoor,
        test_env_id=9,
        file_name="new_colorspace_sharpend_indoor_movloss",
        data_cfg=DataCfg(
            ratio_fov_samples=0.96,
            ratio_back_samples=0.01,
            ratio_front_samples=0.03,
        ),
        wb_project="SemNav-NewColorspace",
    )
    trainer = Trainer(indoor)
    # trainer.train()
    trainer.test()
    # trainer.save_config()
    torch.cuda.empty_cache()

    indoor_noise: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_sharpend",
        env_list=env_list_indoor,
        test_env_id=9,
        file_name="new_colorspace_sharpend_indoor_noise",
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
        "2azQ1b91cZZ_new_colorspace",
        "JeFG25nYj2p_new_colorspace",
        "town01_new_colorspace_train",
        "Vvot9Ly1tCj_new_colorspace",
        "E9uDoFAP3SH_new_colorspace",
        "ur6pFq6Qu1A_new_colorspace",
        "B6ByNegPMKs_new_colorspace",
        "8WUmhLawc2A_new_colorspace",
        "town01_new_colorspace_train",
        "QUCTc6BB5sX_new_colorspace",
        "YFuZgdQ5vWj_new_colorspace",
        "2n8kARJN3HM_new_colorspace",
    ]
    samples_cls_parameters = {
        "ratio_fov_samples": 0.96,
        "ratio_back_samples": 0.01,
        "ratio_front_samples": 0.03,
    }
    data_cfg_list = [
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        DataCfg(
            carla=True, distance_scheme={4: 0.05, 7.5: 0.60, 10: 0.30}, max_train_pairs=15000, **samples_cls_parameters
        ),
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        DataCfg(
            carla=True, distance_scheme={4: 0.05, 7.5: 0.60, 10: 0.30}, max_train_pairs=15000, **samples_cls_parameters
        ),
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
        DataCfg(**samples_cls_parameters),
    ]
    combi: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_sharpend",
        env_list=env_list_combi,
        test_env_id=11,
        file_name="new_colorspace_sharpend_combi",
        data_cfg=data_cfg_list,
        n_visualize=16,
        wb_project="SemNav-NewColorspace",
    )
    # trainer = Trainer(combi)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()

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
        DataCfg(
            carla=True,
            distance_scheme={5: 0.20, 7.5: 0.45, 10: 0.30},
            max_train_pairs=15000,
            **samples_cls_parameters,
            **noise_parameters,
        ),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(
            carla=True,
            distance_scheme={5: 0.20, 7.5: 0.45, 10: 0.30},
            max_train_pairs=15000,
            **samples_cls_parameters,
            **noise_parameters,
        ),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
        DataCfg(**samples_cls_parameters, **noise_parameters),
    ]

    combi_noise: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_sharpend",
        env_list=env_list_combi,
        test_env_id=11,
        file_name="new_colorspace_sharpend_combi_noise",
        data_cfg=data_cfg_list,
        n_visualize=16,
        wb_project="SemNav-NewColorspace",
    )
    # trainer = Trainer(combi_noise)
    # trainer.train()
    # trainer.test()
    # trainer.save_config()
    # torch.cuda.empty_cache()
