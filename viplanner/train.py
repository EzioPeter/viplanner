#!/usr/bin/env python3
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch


@brief      Example train file for VIPlanner
"""

# python
import torch
torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from viplanner.config import TrainCfg, DataCfg
from viplanner.utils.trainer import Trainer


if __name__ == "__main__":
    
    env_list_combi = [
        "2azQ1b91cZZ",
        "JeFG25nYj2p",
        "Vvot9Ly1tCj",
        "town01_more_data_train",
        "ur6pFq6Qu1A",
        "B6ByNegPMKs",        
        "8WUmhLawc2A",
        "town01_more_data_train",
        "2n8kARJN3HM"
    ]
    carla: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem_neg05",
        env_list=env_list_combi,
        test_env_id=8,
        file_name="combi_more_data_neg05",
        data_cfg=DataCfg(
            max_goal_distance=10.0,
        ),
        n_visualize=128,
        wb_project="viplanner-Carla"
    )  
    trainer = Trainer(carla)
    # trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache() 