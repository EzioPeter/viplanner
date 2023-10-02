# python
from viplanner.config import (
    CostMapConfig,
    GeneralCostMapConfig,
    ReconstructionCfg,
    SemCostMapConfig,
)
from viplanner.cost_builder import main

# viplanner
from viplanner.depth_reconstruct import DepthReconstruction

if __name__ == "__main__":
    # ENV 2n8kARJN3HM
    config_reconstruct_town01 = ReconstructionCfg(
        env="town01_new_colorspace_reconstruct",
        voxel_size=0.1,
        high_res_depth=True,
        point_cloud_batch_size=100,
        max_images=1000,
    )
    depth_constructor = DepthReconstruction(config_reconstruct_town01)
    depth_constructor.depth_reconstruction()
    depth_constructor.save_pcd()
    depth_constructor.show_pcd()

    config_cost_town01 = CostMapConfig(
        visualize=False,
        semantics=True,
        geometry=False,
        map_name="tsdf_sem",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/viplanner/imperative_learning/data/town01_new_colorspace_train",
            x_min=-8.05,
            y_min=-8.05,
            x_max=346.22,
            y_max=336.65,
            resolution=0.1,
        ),
        sem_cost_map=SemCostMapConfig(
            ground_height=-0.5,
            obstacle_threshold=0.7,
        ),
    )

    # build cost
    main(config_cost_town01, final_viz=True)

# EoF
