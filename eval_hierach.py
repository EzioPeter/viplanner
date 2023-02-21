#!/usr/bin/env python3

# python
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# imperative-planning-learning
from config import TrainCfg, DataCfg
from utils.trainer import Trainer


if __name__ == "__main__":
    matterport_overfit: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["2n8kARJN3HM", "2n8kARJN3HM"],
        test_env_id=1,
        file_name="_overfit_test",
        hierarchical=True,
    )
    trainer = Trainer(matterport_overfit)
    # set random seed for reproducibility
    torch.manual_seed(trainer._cfg.seed)
    
    # get max steps
    if trainer._cfg.hierarchical:
        step = int(trainer._cfg.epochs / trainer._cfg.hierarchical_step)

    # get dataloader for training
    trainer._load_data(train=False)
    _, test_loader = trainer._get_dataloader(train=False, step=step)    

    # init model
    trainer._load_model()
    
    # test loss buffer
    test_loss = np.zeros((step, 2))
    
    for current_step in range(step):
        # get model parameters
        epoch = trainer._cfg.hierarchical_step * (current_step + 1) - 1
        fov_ratio = 1.0 - (trainer._cfg.hierarchical_back_step_ratio + trainer._cfg.hierarchical_front_step_ratio) * current_step
        front_ratio = trainer._cfg.hierarchical_front_step_ratio * current_step
        back_ratio = trainer._cfg.hierarchical_back_step_ratio * current_step
        model_path = os.path.join(trainer.model_dir, "hierarchical", f"model_ep{epoch}_fov{round(fov_ratio, 3)}_front{round(front_ratio, 3)}_back{round(back_ratio, 3)}.pt")
        # load model at the step
        model_state_dict, best_loss = torch.load(model_path)
        trainer.net.load_state_dict(model_state_dict)
        print("Resume train from {} with loss {}".format(model_path, best_loss))    
        
        test_loss[current_step, 0] = epoch
        test_loss[current_step, 1] = trainer._test_epoch(
            test_loader[0], 
            env_id=0, 
            is_visual=True, 
            fov_angle=trainer.data_generators[0].alpha_fov,
            dataset="test",
        )
    
    # plot test loss 
    plt.figure(figsize=(10, 10))
    plt.plot(test_loss[:, 0], test_loss[:, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Hierarchical Losses")
    plt.savefig(os.path.join(trainer.model_dir, "hierarchical", "hierarchical_test_losses.png"))
    plt.show()
            
# EoF
