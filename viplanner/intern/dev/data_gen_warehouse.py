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
    config_reconstruct_warehouse = ReconstructionCfg(
        env="warehouse_multiple_shelves_without_ppl_ext_sem_space",
        voxel_size=0.05,
        high_res_depth=False,
        point_cloud_batch_size=200,
        max_images=200,
    )
    depth_constructor = DepthReconstruction(config_reconstruct_warehouse)
    depth_constructor.depth_reconstruction()
    depth_constructor.save_pcd()
    depth_constructor.show_pcd()

    config_cost_warehouse = CostMapConfig(
        visualize=False,
        semantics=True,
        geometry=False,
        map_name="tsdf_sem",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/viplanner/imperative_learning/data/warehouse_multiple_shelves_without_ppl_ext_sem_space",
            resolution=0.04,
        ),
        sem_cost_map=SemCostMapConfig(
            ground_height=-0.1,
            obstacle_threshold=0.7,
            robot_height=1.0,
        ),
    )

    # build cost
    main(config_cost_warehouse, final_viz=True)

# EoF
