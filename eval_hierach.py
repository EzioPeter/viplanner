#!/usr/bin/env python3

# python
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# imperative-planning-learning
from config import TrainCfg
from utils.trainer import Trainer


if __name__ == "__main__":
    # load config
    cfg_dir = "/home/pascal/SemNav/imperative_learning/models/plannernet_env2n8kARJN3HM_ep200_inputDepSem_costSem_optimSGD_hierarch_overfit_ratio0.15_oloss0.25_dataDistanceScheme_oWeight0.35"
    train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(cfg_dir, "model.yaml"))
    # train_config.data_cfg.ratio_fov_hard_samples = 1.0
    # train_config.data_cfg.ratio_fov_easy_samples = 0.0
    # train_config.data_cfg.max_train_pairs = 1000
    
    # load trainer and data
    trainer = Trainer(train_config)
    # set random seed for reproducibility
    torch.manual_seed(12)
    # load data and model
    trainer._load_data(train=False)
    trainer._load_model()
    
    # get max steps
    if trainer._cfg.hierarchical:
        step = int(trainer._cfg.epochs / trainer._cfg.hierarchical_step)

    # get dataloader for training
    _, test_loader = trainer._get_dataloader(train=False, step=step-1)  #  0)
    
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
        if os.path.isfile(model_path):
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
        else:
            test_loss = np.delete(test_loss, current_step, axis=0)

    # check for model without hierarchical training setup
    input_domain = "DepSem" if trainer._cfg.sem else "Dep"
    cost_name = "Geom" if trainer._cfg.cost_map_name == "cost_map_geom" else "Sem"
    optim = "SGD" if trainer._cfg.optimizer == "sgd" else "Adam"
    name = f"_{trainer._cfg.file_name}" if trainer._cfg.file_name is not None else ""
    if os.path.isdir(os.path.join("/home/pascal/SemNav/imperative_learning", "models", f"plannernet_env{trainer._cfg.env_list[0]}_ep{trainer._cfg.epochs}_input{input_domain}_cost{cost_name}_optim{optim}{name}")):
        model_path = os.path.join("/home/pascal/SemNav/imperative_learning", "models", f"plannernet_env{trainer._cfg.env_list[0]}_ep{trainer._cfg.epochs}_input{input_domain}_cost{cost_name}_optim{optim}{name}", "model.pt")
        model_state_dict, best_loss = torch.load(model_path)
        trainer.net.load_state_dict(model_state_dict)
        print("Resume train from {} with loss {}".format(model_path, best_loss))    
        
        test_loss_non_hierarch = trainer._test_epoch(
            test_loader[0], 
            env_id=0, 
            is_visual=True, 
            fov_angle=trainer.data_generators[0].alpha_fov,
            dataset="test",
        )
    else:
        test_loss_non_hierarch = None
    
    # plot test loss 
    plt.figure(figsize=(10, 10))
    plt.plot(test_loss[:, 0], test_loss[:, 1], label="Hierarchical", color="blue")
    if test_loss_non_hierarch is not None:
        plt.scatter(trainer._cfg.epochs, test_loss_non_hierarch, label="Non-Hierarchical", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Hierarchical Losses")
    plt.legend()
    plt.savefig(os.path.join(trainer.model_dir_hierarch, "hierarchical_test_losses.png"))
    plt.show()
     
# EoF
