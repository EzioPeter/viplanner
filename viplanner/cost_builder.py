#!/usr/bin python3

"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      cost map builder for imperative learning
"""

# python
import os
import numpy as np
import yaml

# imperative-cost-map
from viplanner.config import CostMapConfig
from viplanner.cost_maps import SemCostMap, TsdfCostMap, CostMapPCD


def main(cfg: CostMapConfig, final_viz: bool = True):
    
    assert any([cfg.semantics, cfg.geometry]), "no cost map type selected"
    
    # create semantic cost map
    if cfg.semantics:
        print("============ Creating Semantic Map from cloud ===============")
        sem_cost_map = SemCostMap(cfg.general, cfg.sem_cost_map, visualize=cfg.visualize)
        sem_cost_map.pcd_init()
        data, coord = sem_cost_map.create_costmap()
    # create tsdf cost map
    elif cfg.geometry:
        print("============== Creating tsdf Map from cloud =================")
        tsdf_cost_map = TsdfCostMap(cfg.general, cfg.tsdf_cost_map)
        tsdf_cost_map.ReadPointFromFile()
        data, coord = tsdf_cost_map.CreateTSDFMap()
        tsdf_cost_map.VizCloud(tsdf_cost_map.obs_pcd) if cfg.visualize else None
    else:
        raise ValueError("no cost map type selected")
    
    # construct final cost map as pcd and save parameters
    print("======== Generate and Save costmap as Point-Cloud ===========")
    cost_mapper = CostMapPCD()
    cost_mapper.DirectLoadMap(data, coord, [cfg.general.resolution, cfg.general.clear_dist])
    cost_mapper.SaveTSDFMap(cfg.general.root_path, cfg.map_name)
    if final_viz:
        cost_mapper.ShowTSDFMap(cost_map=True)
    
    # save config parameters
    yaml_path = os.path.join(cfg.general.root_path, "maps", "params", f"config_{cfg.map_name}.yaml")
    with open(yaml_path, 'w+') as file:
        yaml.dump(vars(cfg), file, allow_unicode=True, default_flow_style=False)
    return


if __name__ == "__main__":
    cfg = CostMapConfig()
    main(cfg)
    
# EoF
