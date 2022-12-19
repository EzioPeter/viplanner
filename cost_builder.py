#!/usr/bin python3

"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      cost map builder for imperative learning
"""

# python
import os
import numpy as np
from scipy import ndimage
import math

# imperative-cost-map
from config import CostMapConfig
from cost_maps import SemCostMap, TsdfCostMap, CostToPcd


def main(cfg: CostMapConfig):
    
    assert any([cfg.semantics, cfg.geometry]), "no cost map type selected"
    
    # create semantic cost map
    if cfg.semantics:
        print("============ Creating Semantic Map from cloud ===============")
        sem_cost_map = SemCostMap(cfg.general, cfg.sem_cost_map, visualize=cfg.visualize)
        sem_cost_map.pcd_init()
        data_sem, coord_sem = sem_cost_map.create_costmap()
    
    # create tsdf cost map
    if cfg.geometry:
        print("============== Creating tsdf Map from cloud =================")
        tsdf_cost_map = TsdfCostMap(cfg.general, cfg.tsdf_cost_map)
        tsdf_cost_map.ReadPointFromFile()
        data_tsdf, coord_tsdf = tsdf_cost_map.CreateTSDFMap()
        tsdf_cost_map.VizCloud(tsdf_cost_map.obs_pcd) if cfg.visualize else None
    
    # combine cost maps if both are selected
    if all([cfg.semantics, cfg.geometry]):
        # check coords, params 
        assert (np.round(coord_sem, decimals=5) == np.round(coord_tsdf, decimals=5)).all(), f"Cost-Map offsets do not match (caused by filtering operations)\n {np.round(coord_sem, decimals=5)} vs {np.round(coord_tsdf, decimals=5)}"
        assert data_sem[0].shape == data_tsdf[0].shape, "Cost-Map shapes do not match"
        
        # combine data and apply additional smoothing
        cost_map = data_sem[0] + data_tsdf[0]
                
        if cfg.visualize:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(data_sem[0])
            axs[1].imshow(data_tsdf[0])
            plt.show()
        
        # rename parameter
        data = [cost_map, data_tsdf[1], data_tsdf[2]]  # TODO: when height-map implemented, check if both are equal before just passing one
        coord = coord_sem
        data_sem = data_tsdf = None  # free memory
    else:
        # rename parameters
        data = data_sem if cfg.semantics else data_tsdf
        coord = coord_sem if cfg.semantics else coord_tsdf
    
    # construct final cost map as pcd and save parameters
    print("======== Generate and Save costmap as Point-Cloud ===========")
    cost_mapper = CostToPcd()
    cost_mapper.DirectLoadMap(data, coord, [cfg.general.resolution, cfg.general.clear_dist])
    cost_mapper.SaveTSDFMap(cfg.general.root_path, cfg.map_name)
    cost_mapper.ShowTSDFMap(cost_map=True)
    
    return


if __name__ == "__main__":
    cfg = CostMapConfig()
    main(cfg)
    
# EoF
