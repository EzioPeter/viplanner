#!/usr/bin/env python3

# python
import os
import yaml
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import wandb  # logging
from typing import Tuple, List

torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from config import TrainCfg, DataCfg
from plannernet import AutoEncoder, DualAutoEncoder
from utils.torchutil import EarlyStopScheduler, count_parameters
from dataset import PlannerData, PlannerDataGenerator, PlannerDataOld, MultiEpochsDataLoader
from traj_cost_opt import TrajCost, TrajViz


def train(
    cfg: TrainCfg, 
    loader: Data.DataLoader,
    net: torch.nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    traj_cost: TrajCost,
    env_id: int,
    sem: bool
) -> float:
    """ Train network with single or multiple domains"""
    train_loss, batches = 0, len(loader)
    enumerater = tqdm.tqdm(enumerate(loader))

    for batch_idx, inputs in enumerater:
        odom  = inputs[2].cuda(cfg.gpu_id) if not cfg.old_plannerdata else inputs[1].cuda(cfg.gpu_id)
        goal  = inputs[3].cuda(cfg.gpu_id) if not cfg.old_plannerdata else inputs[2].cuda(cfg.gpu_id)
        optimizer.zero_grad()
        
        if sem:
            depth_image = inputs[0].cuda(cfg.gpu_id)
            sem_image = inputs[1].cuda(cfg.gpu_id)
            preds, fear = net(depth_image, sem_image, goal)
        else: 
            image = inputs[0].cuda(cfg.gpu_id)
            preds, fear = net(image, goal)

        # flip y axis for augmented samples  (clone necessary due to inplace operation that otherwise leads to error in backprop)
        preds_flip = torch.clone(preds)
        preds_flip[inputs[6], :, 1] = preds_flip[inputs[6], :, 1] * -1
        goal_flip = torch.clone(goal)
        goal_flip[inputs[6], 1] = goal_flip[inputs[6], 1] * -1
            
        log_step = batch_idx + epoch*batches
        loss, _ = MapObsLoss(cfg, preds_flip, fear, traj_cost, odom, goal_flip, pair_difficult=inputs[4], pair_outside=inputs[5], log_step=log_step)
        wandb.log({"train_loss_step": loss}, step=log_step)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        enumerater.set_description(f"Epoch: {epoch} in Env: ({env_id+1}/{len(cfg.env_list)-1}) - train loss:{round(train_loss/(batch_idx+1), 4)} on {batch_idx}/{batches}")
    return train_loss/(batch_idx+1)

def performance(
    cfg: TrainCfg,
    loader,
    net,
    traj_cost: TrajCost,
    traj_viz: TrajViz,
    epoch: int = 0,
    is_visual=False,
    sem: bool = False,
    fov_angle: float = 90.0,
    dataset: str = "val",
) -> float:
    """Evaluate network with single or multiple domains"""
    test_loss = 0
    num_batches = len(loader)
    preds_viz = []; wp_viz = []

    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            odom  = inputs[2].cuda(cfg.gpu_id) if not cfg.old_plannerdata else inputs[1].cuda(cfg.gpu_id)
            goal  = inputs[3].cuda(cfg.gpu_id) if not cfg.old_plannerdata else inputs[2].cuda(cfg.gpu_id)
            
            if sem:
                image = inputs[0].cuda(cfg.gpu_id)  # depth
                sem_image = inputs[1].cuda(cfg.gpu_id)  # semantic
                preds, fear = net(image, sem_image, goal)
            else:
                image = inputs[0].cuda(cfg.gpu_id)
                preds, fear = net(image, goal)

            # flip y axis for augmented samples
            preds[inputs[6], :, 1] = preds[inputs[6], :, 1] * -1
            goal[inputs[6], 1] = goal[inputs[6], 1] * -1
            
            log_step = epoch * num_batches + batch_idx
            loss, waypoints = MapObsLoss(cfg, preds, fear, traj_cost, odom, goal, log_step=log_step, pair_difficult=inputs[4], pair_outside=inputs[5], dataset=dataset)
            
            if cfg.training:
                wandb.log({f"{dataset}_loss_step": loss}, step=log_step)

            test_loss += loss.item()

            if is_visual and len(preds_viz) < cfg.n_visualize:
                if batch_idx == 0:
                    image_viz = image
                    odom_viz = odom
                    goal_viz = goal
                    fear_viz = fear
                    augment_viz = inputs[6]
                else:
                    image_viz   = torch.cat((image_viz, image), dim=0)
                    odom_viz    = torch.cat((odom_viz, odom),   dim=0)
                    goal_viz    = torch.cat((goal_viz, goal),   dim=0)
                    fear_viz    = torch.cat((fear_viz, fear),   dim=0)
                    augment_viz = torch.cat((augment_viz, inputs[6]), dim=0)
                preds_viz.extend(preds.tolist())
                wp_viz.extend(waypoints.tolist())

        if is_visual:
            max_n = min(len(wp_viz), cfg.n_visualize)
            preds_viz   = preds_viz[:max_n]
            wp_viz      = wp_viz[:max_n]
            odom_viz    = odom_viz[:max_n].cpu()
            goal_viz    = goal_viz[:max_n].cpu()
            fear_viz    = fear_viz[:max_n, :].cpu()
            image_viz   = image_viz[:max_n].cpu()
            augment_viz = augment_viz[:max_n].cpu()
            # visual trajectory and images
            traj_viz.VizTrajectory(preds_viz, wp_viz, odom_viz, goal_viz, fear_viz, fov_angle=fov_angle, augment_viz=augment_viz)
            # traj_viz.VizImages(preds_viz, wp_viz, odom_viz, goal_viz, fear_viz, image_viz)
    return test_loss/(batch_idx+1)

def MapObsLoss(
    cfg: TrainCfg,
    preds: torch.Tensor,
    fear: torch.Tensor,
    traj_cost: TrajCost,
    odom: torch.Tensor,
    goal: torch.Tensor,
    log_step: int,
    pair_difficult: List[bool] = [False],
    pair_outside: List[bool] = [False],
    step: float = 0.1,
    dataset: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor]:
    waypoints = traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=step)
    loss1, fear_labels = traj_cost.CostofTraj(
        waypoints,
        odom,
        goal,
        log_step,
        ahead_dist=cfg.fear_ahead_dist,
        pair_difficult=pair_difficult,
        pair_outside=pair_outside,
        dataset=dataset,
    )
    loss2 = nn.BCELoss()(fear, fear_labels)
    return loss1+loss2, waypoints

def run_train(
    cfg: TrainCfg,
    model_path: str,
    data_dir: str,
    transform: transforms.Compose,
):
    """Load Model"""
    best_loss = float('Inf')
    if cfg.sem:
        net = DualAutoEncoder(cfg.in_channel, cfg.knodes)
    else:
        net = AutoEncoder(cfg.in_channel, cfg.knodes)

    if cfg.resume:
        model_state_dict, best_loss = torch.load(model_path)
        net.load_state_dict(model_state_dict)
        print("Resume train from {} with loss {}".format(model_path, best_loss))

    
    assert torch.cuda.is_available(), "Code requires GPU"
    print("Available GPU list: {}".format(list(range(torch.cuda.device_count()))))
    print("Runnin on GPU: {}".format(cfg.gpu_id))
    net = net.cuda(cfg.gpu_id)

    print('number of parameters:', count_parameters(net))
    
    """Configure Optimizer and Scheduler"""
    if cfg.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.w_decay)
    elif cfg.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.w_decay)
    else:
        raise KeyError("Optimizer {} not supported".format(cfg.optimizer))
    scheduler = EarlyStopScheduler(optimizer, factor=cfg.factor, verbose=True, min_lr=cfg.min_lr, patience=cfg.patience)
    
    """Load Data"""
    train_loader_list = []
    val_loader_list  = []
    traj_cost_list = []
    traj_viz_list = []
    
    for idx, env_name in enumerate(cfg.env_list):
        if cfg.training == True and idx == cfg.test_env_id:
            continue
            
        data_path = os.path.join(*[data_dir, env_name])
        traj_cost, traj_viz = get_cost_and_viz(cfg, data_path)
        
        # create dataset
        if not cfg.old_plannerdata:
            train_data = PlannerData(
                cfg=cfg.data_cfg,
                transform=transform,
                semantics=cfg.sem,
            )
            val_data = PlannerData(
                cfg=cfg.data_cfg,
                transform=transform,
                semantics=cfg.sem
            )
            generator = PlannerDataGenerator(
                cfg=cfg.data_cfg,
                root=data_path,
                semantics=cfg.sem,
                tsdf_map=traj_cost.tsdf_map  # trajectory cost class
            )
            generator.split_samples(
                train_dataset=train_data,
                test_dataset=val_data,
                generate_split=True
            )
            generator = None
        else:
            train_data = PlannerDataOld(
                root=data_path,
                train=True,
                transform=transform,
                max_depth=cfg.data_cfg.max_depth,
                )
            val_data = PlannerDataOld(
                root=data_path,
                train=False,
                transform=transform,
                max_depth=cfg.data_cfg.max_depth,
                )
        if cfg.multi_epoch_dataloader:
            train_loader = MultiEpochsDataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
            val_loader = MultiEpochsDataLoader(val_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
        else:
            train_loader = Data.DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
            val_loader = Data.DataLoader(dataset=val_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
        
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
        traj_cost_list.append(traj_cost)
        traj_viz_list.append(traj_viz)

    print("Data Loading Completed!")

    """ Training with mutilple envs """
    if cfg.training == True:
        # wandb watch model
        wandb.watch(net)
        
        for epoch in range(cfg.epochs):
            train_loss = 0; val_loss = 0
            for i in range(len(train_loader_list)):
                train_loss += train(cfg, train_loader_list[i], net, optimizer, epoch, traj_cost_list[i], env_id=i, sem=cfg.sem)
                val_loss += performance(cfg, val_loader_list[i], net, traj_cost_list[i], traj_viz_list[i], sem=cfg.sem, epoch=epoch)

            train_loss /= len(train_loader_list)
            val_loss /= len(train_loader_list)
            
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
            
            # if val_loss < best_loss:
            if val_loss < best_loss:
                print("Save model of epoch %d"%(epoch))
                torch.save((net.state_dict(), val_loss), model_path)
                best_loss = val_loss
                print("Current val loss: %.4f"%(best_loss))

            if scheduler.step(val_loss):
                print('Early Stopping!')
                break
                
    return best_loss

def run_testing(
    cfg: TrainCfg,
    model_path: str,
    data_dir: str,
    transform: transforms.Compose,
) -> float:
    print("Testing")
    test_env_id = cfg.test_env_id

    # set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    
    # for nicer plots, set minimum start-goal distance to higher value
    if not os.getenv('EXPERIMENT_DIRECTORY'):
        cfg.data_cfg.min_goal_distance = 2.0

    if cfg.sem:
        net_test = DualAutoEncoder(cfg.in_channel, cfg.knodes)
    else:
        net_test = AutoEncoder(cfg.in_channel, cfg.knodes)
    
    model_state_dict, _ = torch.load(model_path)
    net_test.load_state_dict(model_state_dict)        
    if torch.cuda.is_available():
        net_test = net_test.cuda(cfg.gpu_id)

    test_path = os.path.join(*[data_dir, cfg.env_list[test_env_id]])

    traj_cost, traj_viz = get_cost_and_viz(cfg, test_path)
    
    if not cfg.old_plannerdata:
        test_data = PlannerData(
            cfg=cfg.data_cfg,
            transform=transform,
            semantics=cfg.sem
        )
        generator = PlannerDataGenerator(
            cfg=cfg.data_cfg,
            root=test_path,
            semantics=cfg.sem,
            tsdf_map=traj_cost.tsdf_map  # trajectory cost class
        )
        generator.split_samples(
            test_dataset=test_data,
            generate_split=False
        )
        generator = None
    else:
        test_data = PlannerDataOld(
            root=test_path,
            train=False,
            transform=transform,
            max_depth=cfg.data_cfg.max_depth,
        )        
    
    if cfg.multi_epoch_dataloader:
        test_loader = MultiEpochsDataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
    else:
        test_loader = Data.DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.num_workers)
    
    test_loss = performance(
        cfg, 
        test_loader, 
        net_test, 
        traj_cost, 
        traj_viz, 
        is_visual = False if os.getenv('EXPERIMENT_DIRECTORY') else True, 
        sem=cfg.sem,
        fov_angle=test_data.fov_angle,
        dataset="test",
    )
    return test_loss

def get_cost_and_viz(
    cfg: TrainCfg,
    data_path: str
    ) -> Tuple[TrajCost, TrajViz]:
    """Get trajectory cost and visualization for the different datasets""" 
        
    # Load Map and Trajectory Class
    traj_cost = TrajCost(
        cfg.gpu_id,
        sensorOffsetX=cfg.sensor_offsetX_ANYmal,
        weight_difficult=cfg.weight_samples_difficult,
        weight_outside=cfg.weight_samples_high_cost,
        log_data=cfg.training
    )
    traj_cost.SetMap(
        data_path,
        cfg.cost_map_name,
    )
    traj_viz = TrajViz(
        data_path,
        cfg.cost_map_name,
        sensorOffsetX=cfg.sensor_offsetX_ANYmal,
        cameraTilt=cfg.camera_tilt
    )

    return traj_cost, traj_viz


def model_train(cfg: TrainCfg) -> None:
    print(cfg)
    
    # set model save/load path
    model_dir = os.path.join(os.getenv('EXPERIMENT_DIRECTORY', "/home/pascal/SemNav/imperative_learning"), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, cfg._get_model_save())
    
    # set data root directory  --> to make it work on euler cluster
    data_dir = os.path.join(os.getenv('EXPERIMENT_DIRECTORY', "/home/pascal/SemNav/imperative_learning"), "data")
    
    transform = transforms.Compose([
        transforms.Resize((cfg.img_input_size)),
        transforms.ToTensor()])

    # logging
    if cfg.training:
        os.environ["WANDB_API_KEY"] = cfg.wb_api_key
        os.environ["WANDB_MODE"] = "offline" if os.getenv('EXPERIMENT_DIRECTORY') else "online"
        dir_path = os.path.join(os.getenv('EXPERIMENT_DIRECTORY', "/home/pascal/SemNav/imperative_learning"), "logs")
        os.makedirs(dir_path, exist_ok=True)
        
        wandb.init(
            project=cfg.wb_project,
            entity=cfg.wb_entity,
            name=cfg._get_model_save()[:-3],
            config=cfg.__dict__,
            dir=dir_path
        )
    
    """RUN TRAINING"""
    if cfg.training:
        best_loss = run_train(cfg, model_path, data_dir, transform)
        torch.cuda.empty_cache()
    else:
        best_loss = "Inf"
    
    """RUN Testing """
    test_loss = run_testing(cfg, model_path, data_dir, transform)

    if cfg.training:
        print('val_loss: %.2f, test_loss, %.4f'%(best_loss, test_loss))
        """ Save config and loss to file"""
        path, _ = os.path.splitext(model_path)
        yaml_path = path + ".yaml"
        print(f"Save config and loss to {yaml_path} file")
        
        loss_dict = {"val_loss": best_loss, "test_loss": test_loss}
        save_dict = {"config": vars(cfg), "loss": loss_dict}
        
        # dump yaml
        with open(yaml_path, 'w+') as file:
            yaml.dump(save_dict, file, allow_unicode=True, default_flow_style=False)

        # logging
        wandb.finish()
    else:
        print('test_loss, %.4f'%(test_loss))
    return


if __name__ == "__main__":
    # Arguements  
    matterport_overfit: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["2n8kARJN3HM", "2n8kARJN3HM"],
        test_env_id=1
        file_name="_overfit",
    )
    model_train(matterport_overfit)
    torch.cuda.empty_cache()
    
    matterport_sem: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
    )  
    # model_train(matterport_sem)
    # torch.cuda.empty_cache()
            
    carla: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["town01", "town01"],
        test_env_id=1,
        data_cfg=DataCfg(
            max_goal_distance=10.0,
            max_depth=15,
        ),
        n_visualize=128,
        wb_project="SemNav-Carla"
    )  
    # model_train(carla)   
    # torch.cuda.empty_cache() 

    carla_obscost: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=["town01", "town01"],
        test_env_id=1,
        file_name="_obscostheight_05",
        data_cfg=DataCfg(
            max_goal_distance=10.0,
            max_depth=15,
            obs_cost_height=0.5,
            ratio_hard=0.5,
            ratio_easy=0.4,
            ratio_outside=0.1,
        ),
        n_visualize=400,
        wb_project="SemNav-Carla"
    )      
    # model_train(carla_obscost)

# EoF
