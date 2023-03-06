from depth_reconstruct import DepthReconstruction
from cost_builder import main
from config import ReconstructionCfg, CostMapConfig, GeneralCostMapConfig

def reconstruct(cfg: ReconstructionCfg):
    depth_constructor = DepthReconstruction(cfg)
    depth_constructor.depth_reconstruction()
    depth_constructor.save_pcd()
    return


if __name__ == "__main__":
    # ENV 2n8kARJN3HM
    config_reconstruct_2n8kARJN3HM = ReconstructionCfg(
        env= "2n8kARJN3HM",
    )
    reconstruct(config_reconstruct_2n8kARJN3HM)
    config_cost_2n8kARJN3HM = CostMapConfig(
        visualize=False,
        semantics=True,
        geometry=False,
        map_name="cost_map_sem",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/2n8kARJN3HM"
        )
    )
    main(config_cost_2n8kARJN3HM, final_viz = False)
    config_cost_2n8kARJN3HM = CostMapConfig(
        visualize=False,
        semantics=False,
        geometry=True,
        map_name="cost_map_geom",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/2n8kARJN3HM"
        )
    )
    main(config_cost_2n8kARJN3HM, final_viz = False)    
    
    # ENV 2azQ1b91cZZ
    config_reconstruct_2azQ1b91cZZ = ReconstructionCfg(
        env= "2azQ1b91cZZ",
    )
    reconstruct(config_reconstruct_2azQ1b91cZZ)
    config_cost_2azQ1b91cZZ = CostMapConfig(
        visualize=False,
        semantics=True,
        geometry=False,
        map_name="cost_map_sem",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/2azQ1b91cZZ"
        )
    )
    main(config_cost_2azQ1b91cZZ, final_viz = False)
    config_cost_2azQ1b91cZZ = CostMapConfig(
        visualize=False,
        semantics=False,
        geometry=True,
        map_name="cost_map_geom",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/2azQ1b91cZZ"
        )
    )
    main(config_cost_2azQ1b91cZZ, final_viz = False)   
    
    # ENV JeFG25nYj2p
    config_reconstruct_JeFG25nYj2p = ReconstructionCfg(
        env= "JeFG25nYj2p",
    )
    reconstruct(config_reconstruct_JeFG25nYj2p)
    config_cost_JeFG25nYj2p = CostMapConfig(
        visualize=False,
        semantics=True,
        geometry=False,
        map_name="cost_map_sem",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/JeFG25nYj2p"
        )
    )
    main(config_cost_JeFG25nYj2p, final_viz = False)
    config_cost_JeFG25nYj2p = CostMapConfig(
        visualize=False,
        semantics=False,
        geometry=True,
        map_name="cost_map_geom",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/JeFG25nYj2p"
        )
    )
    main(config_cost_JeFG25nYj2p, final_viz = False)   
    
    # ENV Vvot9Ly1tCj
    config_reconstruct_Vvot9Ly1tCj = ReconstructionCfg(
        env= "Vvot9Ly1tCj",
    )
    reconstruct(config_reconstruct_Vvot9Ly1tCj)
    config_cost_Vvot9Ly1tCj = CostMapConfig(
        visualize=False,
        semantics=True,
        geometry=False,
        map_name="cost_map_sem",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/Vvot9Ly1tCj"
        )
    )
    main(config_cost_Vvot9Ly1tCj, final_viz = False)
    config_cost_Vvot9Ly1tCj = CostMapConfig(
        visualize=False,
        semantics=False,
        geometry=True,
        map_name="cost_map_geom",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/Vvot9Ly1tCj"
        )
    )
    main(config_cost_Vvot9Ly1tCj, final_viz = False)   
    
    # ENV ur6pFq6Qu1A
    config_reconstruct_ur6pFq6Qu1A = ReconstructionCfg(
        env= "ur6pFq6Qu1A",
    )
    reconstruct(config_reconstruct_ur6pFq6Qu1A)
    config_cost_ur6pFq6Qu1A = CostMapConfig(
        visualize=False,
        semantics=True,
        geometry=False,
        map_name="cost_map_sem",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/ur6pFq6Qu1A"
        )
    )
    main(config_cost_ur6pFq6Qu1A, final_viz = False)
    config_cost_ur6pFq6Qu1A = CostMapConfig(
        visualize=False,
        semantics=False,
        geometry=True,
        map_name="cost_map_geom",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/ur6pFq6Qu1A"
        )
    )
    main(config_cost_ur6pFq6Qu1A, final_viz = False)   
    
    # ENV B6ByNegPMKs
    config_reconstruct_B6ByNegPMKs = ReconstructionCfg(
        env= "B6ByNegPMKs",
    )
    reconstruct(config_reconstruct_B6ByNegPMKs)
    config_cost_B6ByNegPMKs = CostMapConfig(
        visualize=False,
        semantics=True,
        geometry=False,
        map_name="cost_map_sem",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/B6ByNegPMKs"
        )
    )
    main(config_cost_B6ByNegPMKs, final_viz = False)
    config_cost_B6ByNegPMKs = CostMapConfig(
        visualize=False,
        semantics=False,
        geometry=True,
        map_name="cost_map_geom",
        general=GeneralCostMapConfig(
            root_path="/home/pascal/SemNav/imperative_learning/data/B6ByNegPMKs"
        )
    )
    main(config_cost_B6ByNegPMKs, final_viz = False)   

# EoF
