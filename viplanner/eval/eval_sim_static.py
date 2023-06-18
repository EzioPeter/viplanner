#!/usr/bin/env python3

# python
import os
import torch
import numpy as np
import argparse
from typing import List, Optional

# imperative-planning-learning
from viplanner.config import TrainCfg
from viplanner.utils.trainer import Trainer
from viplanner.utils.eval_utils import BaseEvaluator
from viplanner.traj_cost_opt import TrajCost, TrajOpt

class SimEvaluator(BaseEvaluator):
    
    debug: bool = False
    
    def __init__(self, tolerance: float = 0.5, environment: str = "2n8kARJN3HM") -> None:
        # init base class
        super().__init__(tolerance)        
        # set random seed for reproducibility
        torch.manual_seed(12)
        # environment
        self.environment = environment
        return

    def run(self, model_dirs: List[str], model_names: Optional[List[str]] = None, use_prev_results: bool = True):
        if use_prev_results:
            length_goal_list = []
            length_path_list = []
            goal_distance_list = []
            obstacle_loss_list = []
            obstacle_max_loss_list = []
            data_dir = os.path.join("/home/pascal/SemNav/imperative_learning/data", self.environment)
            try:
                for model_dir in model_dirs:
                    _, model_name = os.path.split(model_dir)
                    eval_dir = os.path.join(data_dir, f"eval_{model_name}")
                    
                    length_goal_list.append(np.loadtxt(os.path.join(eval_dir, "length_goal.txt")))
                    length_path_list.append(np.loadtxt(os.path.join(eval_dir, "length_path.txt")))
                    goal_distance_list.append(np.loadtxt(os.path.join(eval_dir, "goal_distances.txt")))
                    obstacle_loss_list.append(np.loadtxt(os.path.join(eval_dir, "loss_obstacles.txt")))
                    obstacle_max_loss_list.append(np.loadtxt(os.path.join(eval_dir, "loss_max_obstacles.txt")))
                success = True
                self._use_cost_map = True
            except FileNotFoundError:
                print("[Warning] No previous results found, running evaluation...")
                success = False
        
        if not use_prev_results or not success:
            # load config
            train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(model_dirs[0], "model.yaml"))
            train_config.env_list = [self.environment]
            train_config.test_env_id = 0
            
            # FOR CARLA: NEW DISTANCE SCHEME
            train_config.data_cfg.obs_cost_height = 0.5
            train_config.data_cfg.distance_scheme = {5: 0.10, 7.5: 0.40, 10: 0.30, 15: 0.15}
            # load trainer and data
            self.trainer = Trainer(train_config)        
            # load data
            self.setup()
            
            # run model
            length_goal_list = []
            length_path_list = []
            goal_distance_list = []
            obstacle_loss_list = []

            for model_dir in model_dirs:
                # reset
                self.reset()
                del self.trainer.net
                # load new config
                self.trainer.model_path = os.path.join(model_dir, "model.pt")
                train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(model_dir, "model.yaml"))
                train_config.env_list = [self.environment]
                train_config.test_env_id = 0
                self.trainer._cfg = train_config
                # load model
                self.trainer._load_model(resume=True)
                # run evaluation
                self.run_eval()
                
                length_goal_list.append(self.length_goal.copy())
                length_path_list.append(self.length_path.copy())
                goal_distance_list.append(self.goal_distances.copy())
                obstacle_loss_list.append(self.loss_obstacles.copy())
        
        self.plt_comparison(length_goal_list, length_path_list, goal_distance_list, model_dirs, data_dir, obstacle_loss_list, model_names)
        return
    
    def setup(self):
        # get dataloader for training
        self.trainer._load_data(train=False)
        _, test_loader = self.trainer._get_dataloader(train=False, allow_augmentation=False)
        self.test_loader = test_loader[0]
        
        # set cost map
        self._use_cost_map = True
        self._traj_cost: TrajCost = self.trainer.data_traj_cost[0]

        # create buffers
        nbr_samples = len(self.test_loader) * self.trainer._cfg.batch_size
        self.set_nbr_paths(nbr_samples)
        return
   
    def create_buffers(self) -> None:
        super().create_buffers()
        self.loss_max_obstacles = np.zeros((self.nbr_paths))
        return
    
    def eval_statistics(self) -> None:
        super().eval_statistics()
        path_success_obs = np.sum(self.loss_max_obstacles > 0.3) / self.nbr_paths
        
        print(
                f"\nPath success obs loss:        {path_success_obs}"
            )
        return
        
    def run_eval(self):
        pred_counter = 0
        
        with torch.no_grad():
            for inputs in self.test_loader:
                odom  = inputs[2].cuda(self.trainer._cfg.gpu_id)
                goal  = inputs[3].cuda(self.trainer._cfg.gpu_id)
                
                if self.trainer._cfg.sem or self.trainer._cfg.rgb:
                    image = inputs[0].cuda(self.trainer._cfg.gpu_id)  # depth
                    sem_rgb_image = inputs[1].cuda(self.trainer._cfg.gpu_id)  # semantic
                    preds, fear = self.trainer.net(image, sem_rgb_image, goal)
                else:
                    image = inputs[0].cuda(self.trainer._cfg.gpu_id)
                    preds, fear = self.trainer.net(image, goal)

                # flip y axis for augmented samples
                preds[inputs[4], :, 1] = preds[inputs[4], :, 1] * -1
                goal[inputs[4], 1] = goal[inputs[4], 1] * -1
                
                # filter paths with high fear
                fear_selection = (fear < 0.5).squeeze()
                preds = preds[fear_selection]
                goal  = goal[fear_selection]
                odom  = odom[fear_selection]
                fear = fear[fear_selection]
                
                if len(preds.shape) == 4:
                    # squeeze
                    preds = preds.squeeze(0)
                    goal  = goal.squeeze(0)
                    odom  = odom.squeeze(0)
                    fear = fear.squeeze(0)
                    
                # optimize
                waypoints = self._traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=0.1)
                waypoints_world = self._traj_cost.TransformPoints(odom, waypoints).tensor().cpu().numpy()
                goal_world = self._traj_cost.TransformPoints(odom, goal[:, None, :3]).tensor().cpu().numpy()
                # evaluate
                self.goal_distances[pred_counter:pred_counter+len(goal)] = np.linalg.norm(waypoints_world[:, -1, :2] - goal_world[:, 0, :2], axis=1)
                self.length_path[pred_counter:pred_counter+len(goal)]    = np.sum(np.linalg.norm(waypoints_world[:, 1:, :2] - waypoints_world[:, :-1, :2], axis=2), axis=1)
                self.length_goal[pred_counter:pred_counter+len(goal)]    = np.linalg.norm(goal_world[:, 0, :2] - odom[:, :2].cpu().numpy(), axis=1)
                mean_loss, max_loss = self._traj_cost.obs_cost_eval(odom, waypoints)
                self.loss_obstacles[pred_counter:pred_counter+len(goal)] = mean_loss.cpu().numpy()
                self.loss_max_obstacles[pred_counter:pred_counter+len(goal)] = max_loss.cpu().numpy()
                pred_counter += len(goal)

                if self.debug:
                    path_diff = self.length_path[pred_counter:pred_counter+len(goal)] - self.length_goal[pred_counter:pred_counter+len(goal)]
                    largest_indices = torch.tensor(np.argsort(path_diff)[-30:])
                    self.trainer.data_traj_viz[0].VizTrajectory(preds.cpu()[largest_indices], waypoints.cpu()[largest_indices], odom.cpu()[largest_indices], goal.cpu()[largest_indices], fear.cpu()[largest_indices], fov_angle=self.trainer.data_generators[0].alpha_fov, augment_viz=inputs[4].cpu()[largest_indices])

        # crop buffers
        self.goal_distances = self.goal_distances[:pred_counter]
        self.length_path = self.length_path[:pred_counter]
        self.length_goal = self.length_goal[:pred_counter]
        self.loss_obstacles = self.loss_obstacles[:pred_counter]
        self.loss_max_obstacles = self.loss_max_obstacles[:pred_counter]
        
        # sort values
        sort_indices = np.argsort(self.length_goal)
        self.length_goal = self.length_goal[sort_indices]
        self.length_path = self.length_path[sort_indices]
        self.goal_distances = self.goal_distances[sort_indices]
        self.loss_obstacles = self.loss_obstacles[sort_indices]
        self.loss_max_obstacles = self.loss_max_obstacles[sort_indices] 
        
        # make directory and save data
        _, model_name = os.path.split(self.trainer._cfg.curr_model_dir)
        data_dir = os.path.join(self.trainer._cfg.data_dir, self.trainer._cfg.env_list[self.trainer._cfg.test_env_id])
        eval_dir = os.path.join(data_dir, f"eval_{model_name}")
        os.makedirs(eval_dir, exist_ok=True)

        np.savetxt(os.path.join(eval_dir, "length_path.txt"), self.length_path)
        np.savetxt(os.path.join(eval_dir, "length_goal.txt"), self.length_goal)
        np.savetxt(os.path.join(eval_dir, "goal_distances.txt"), self.goal_distances)
        np.savetxt(os.path.join(eval_dir, "loss_obstacles.txt"), self.loss_obstacles)
        np.savetxt(os.path.join(eval_dir, "loss_max_obstacles.txt"), self.loss_max_obstacles)
        
        # plot data
        self.plt_single_model(eval_dir, show=False)
        
        # get statistics
        self.eval_statistics()
        self.save_eval_results(self.trainer._cfg.curr_model_dir, save_name=os.path.split(data_dir)[-1])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Model Eval', description='Evaluate VIPmodels')
    parser.add_argument('-m', '--model_dirs', nargs='+', type=str, help='Path to model directory',
                        default=[
                            "/home/pascal/SemNav/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD_neg05",
                            # "/home/pascal/SemNav/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD_combi_more_data_neg05",
                            "/home/pascal/SemNav/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDep_costSem_optimSGD_depth",
                        ])
    parser.add_argument('-n', '--model_names', nargs='+', type=str, help='Model name',
                    default=[
                        "VIPlanner",
                        "iPlanner",
                    ])
    parser.add_argument('-env', '--environment', type=str, help='Environment name',
                default="2n8kARJN3HM")  # "town01_more_data_train")  # 
    parser.add_argument('--tolerance', type=float, help='Tolerance to the goal to be considered reached',
                        default=0.5)
    args = parser.parse_args()
    print(args)

    evaluator = SimEvaluator(args.tolerance, args.environment)
    evaluator.run(args.model_dirs, args.model_names)
    
# EoF
