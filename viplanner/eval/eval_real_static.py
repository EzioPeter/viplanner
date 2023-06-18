"""
@author: Pascal Roth
@email:  rothpa@student.ethz.ch

@brief:  Evaluation script for real world
"""

# python
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from tqdm import tqdm
import pypose as pp
import torchvision.transforms as transforms
from typing import List
import scipy.spatial.transform as tf

# viplanner
from viplanner.config import TrainCfg, Mask2FormerCfg
from viplanner.utils.trainer import Trainer
from viplanner.utils.m2f_utils import M2FWrapper
from viplanner.traj_cost_opt import TrajOpt, TrajCost, TrajViz
from viplanner.dataset import PlannerDataGenerator
from viplanner.utils.eval_utils import BaseEvaluator

# set random seed for reproducibility
torch.manual_seed(12)
time_threshold = 0.1

ROS_TO_ROBOTICS_MAT = tf.Rotation.from_euler("XYZ", [-90, 0, -90], degrees=True).as_matrix()
CAMERA_FLIP_MAT     = tf.Rotation.from_euler("XYZ", [180, 0, 0],   degrees=True).as_matrix()


class RealWorldEvaluator(BaseEvaluator):
    # TODO: inherit from RealWorldDataHandler 
    
    def __init__(self, args: argparse.Namespace, m2f_cfg: Mask2FormerCfg) -> None:
        """
        Make prediction on real world images and evaluate the generated pathes. Expected args:

        args.model_dirs: str    -- model dirs
        args.data_dir: str      -- data directory with the following structure
                                    - data_dir
                                        - depth
                                            - imageXXX.png
                                        - bgr
                                            - imageXXX.png
                                        - odom_depth.txt        (depth odometry values at different time values (x, y, z, qx, qy, qz, qw, sec, nsec))
                                        - odom_bgr.txt          (bgr   odometry values at different time values (x, y, z, qx, qy, qz, qw, sec, nsec))
                                        - intrinsics_bgr.txt
                                        - intrinsics_depth.txt
        """
        super().__init__(args.tolerance)
        self.args = args
        self.m2f_cfg = m2f_cfg
        
        # load data
        self.K_bgr: np.ndarray = None
        self.K_depth: np.ndarray = None
        self.odom_bgr: np.ndarray = None
        self.odom_depth: np.ndarray = None
        self.depth_img_list: np.ndarray = None
        self.bgr_img_list: np.ndarray = None
        self.odom_depth_pp: pp.LieTensor = None
        self.load_data()
        
        # create buffers and set nbr paths
        self.set_nbr_paths(nbr_paths=(len(self.depth_img_list) - max(self.args.goal_frame_advances)) * len(self.args.goal_frame_advances))
        self.create_buffers()
        
        # run model
        length_goal_list = []
        length_path_list = []
        goal_distance_list = []
        for model_dir in self.args.model_dirs:
            length_goal, length_path, goal_distance = self.run_model(model_dir)
            length_goal_list.append(length_goal)
            length_path_list.append(length_path)
            goal_distance_list.append(goal_distance)
            self.reset()
        
        self.plt_comparison(length_goal_list, length_path_list, goal_distance_list, self.args.model_dirs, self.args.data_dir)

        return

    def load_data(self) -> None:
        # load data timestamps and synchroniize them
        odom_bgr   = np.loadtxt(os.path.join(self.args.data_dir, "odom_bgr.txt"))
        odom_depth = np.loadtxt(os.path.join(self.args.data_dir, "odom_depth.txt"))
        depth_idx, bgr_idx = self.synchronize_data(odom_bgr[:, 7:], odom_depth[:, 7:])
        
        # filter images that are timewise too close to each other    
        timestamp_idx = [0] # Always keep the first timestamp
        prev_timestamp = odom_depth[0, 7:]
        for idx, timestamp in enumerate(odom_depth[1:, 7:]):
            time_diff = (timestamp[0] - prev_timestamp[0]) + (timestamp[1] - prev_timestamp[1]) / 1e9
            if time_diff >= time_threshold:
                timestamp_idx.append(idx+1)
                prev_timestamp = timestamp
        depth_idx = depth_idx[timestamp_idx]
        bgr_idx = bgr_idx[timestamp_idx]
                
        # load intrinsics
        self.K_depth = np.loadtxt(os.path.join(self.args.data_dir, "intrinsics_depth.txt"))
        self.K_bgr   = np.loadtxt(os.path.join(self.args.data_dir, "intrinsics_bgr.txt"))
        
        # reduce data to valid depth images, bgr and odom points
        self.depth_img_list = np.array(sorted(os.listdir(os.path.join(self.args.data_dir, "depth"))))
        self.depth_img_list = self.depth_img_list[depth_idx] 
        self.bgr_img_list   = np.array(sorted(os.listdir(os.path.join(self.args.data_dir, "bgr"))))
        self.bgr_img_list   = self.bgr_img_list[bgr_idx]
        odom_depth     = odom_depth[depth_idx, :7]
        odom_bgr       = odom_bgr[bgr_idx, :7]
        
        # transform rotations of depth and bgr image from ROS camera frame (z-forward) to robotics frame (x-forward)
        odom_depth[:, 3:] = tf.Rotation.from_matrix(tf.Rotation.from_quat(odom_depth[:, 3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT).as_quat()
        odom_bgr[:, 3:] = tf.Rotation.from_matrix(tf.Rotation.from_quat(odom_bgr[:, 3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT @ CAMERA_FLIP_MAT).as_quat()
        odom_depth_pp = pp.SE3(odom_depth)

        self.odom_depth = odom_depth
        self.odom_bgr = odom_bgr
        self.odom_depth_pp = odom_depth_pp
        return
        
    def synchronize_data(self, bgr_timestamp, depth_timestamp, threshold=0.1):
        # get combined timestamp
        bgr_timestamp = bgr_timestamp[:, 0] + bgr_timestamp[:, 1] / 1e9
        depth_timestamp = depth_timestamp[:, 0] + depth_timestamp[:, 1] / 1e9
        
        # Find the index of the closest odometry timestamp for each image timestamp
        idx = np.searchsorted(bgr_timestamp, depth_timestamp)
        # Check if the previous odometry timestamp is closer
        prev_idx = np.clip(idx - 1, 0, len(bgr_timestamp) - 1)
        next_idx = np.clip(idx, 0, len(bgr_timestamp) - 1)
        prev_diff = np.abs(depth_timestamp - bgr_timestamp[prev_idx])
        next_diff = np.abs(depth_timestamp - bgr_timestamp[next_idx])
        use_prev = prev_diff < next_diff
        # Compute the synchronized timestamps as a tuple of (odometry timestamp, index)
        synced_timestamp = np.where(use_prev, bgr_timestamp[prev_idx], bgr_timestamp[next_idx])
        synced_idx = np.where(use_prev, prev_idx, next_idx)
        # Check if the synchronized timestamp is within a threshold
        within_threshold = np.abs(synced_timestamp - depth_timestamp) < threshold
        bgr_idx = synced_idx[within_threshold]
        depth_idx = np.where(within_threshold)[0]
        return depth_idx, bgr_idx

    def run_model(self, model_dir: str) -> None:
        # load config
        train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(model_dir, "model.yaml"))

        # if semantics are used, load M2F model and estimate semantics
        if train_config.sem:
            m2f_wrapper = M2FWrapper(self.m2f_cfg)
            m2f_wrapper.run_on_folder(os.path.join(self.args.data_dir, "bgr"), self.args.debug, run_on_existing_files=False)

        # load trainer, model
        trainer = Trainer(train_config)
        trainer._load_model(resume=True)

        # image transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((train_config.img_input_size), antialias=True),
        ])

        # make directory for visualization results
        if self.args.save_viz:
            os.makedirs(os.path.join(self.args.data_dir, "viz"), exist_ok=True)
            
        # init trajectory optimizer and visualizer
        traj_opt = TrajOpt()
        traj_viz = TrajViz(intrinsics=self.K_depth, cam_resolution=train_config.img_input_size)
        
        # init pixel array
        depth_img = cv2.imread(os.path.join(self.args.data_dir, "depth", self.depth_img_list[0]), cv2.IMREAD_ANYDEPTH)
        depth_img = np.asarray(depth_img)
        x_nums, y_nums = depth_img.shape
        depth_pixel_array = PlannerDataGenerator.compute_pixel_tensor(x_nums, y_nums, self.K_depth)

        # make predictions
        pred_counter = 0
        for idx, img in enumerate(tqdm(self.depth_img_list[:-max(self.args.goal_frame_advances)], desc="VIPlanner Predictions")):
            # load images
            depth_img = cv2.imread(os.path.join(self.args.data_dir, "depth", img), cv2.IMREAD_ANYDEPTH)
            depth_img = depth_img.astype(np.float32) / 1000.0
            depth_img[depth_img > train_config.data_cfg.max_depth] = 0.0
            depth_img[~np.isfinite(depth_img)] = 0.0
            if train_config.sem:
                bgr_sem_img = cv2.imread(os.path.join(self.args.data_dir, "semantics", self.bgr_img_list[idx]))
            elif train_config.rgb:
                bgr_sem_img = cv2.imread(os.path.join(self.args.data_dir, "bgr", self.bgr_img_list[idx]))
                raise NotImplementedError("Before Progressing, check if bgr or rgb input")
            if train_config.sem or train_config.rgb:
                bgr_sem_img = cv2.cvtColor(bgr_sem_img, cv2.COLOR_BGR2RGB)  # since loaded as BGR with cv2

                # warp rgb/sem image onto depth image
                bgr_sem_img_warp = PlannerDataGenerator.compute_overlay(self.odom_depth[idx], self.odom_bgr[idx], depth_img=depth_img, sem_rgb_image=bgr_sem_img, pix_depth_cam_frame=depth_pixel_array, K_sem_rgb=self.K_bgr)
                
                # DEBUG
                if self.args.debug:
                    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    ax1.imshow(depth_img)
                    ax2.imshow(bgr_sem_img)
                    ax3.imshow(depth_img)
                    ax3.imshow(bgr_sem_img_warp / 255, alpha=0.5)
                    plt.show()
                
            # select goal
            goal_frames = np.array(self.args.goal_frame_advances) + idx
            goals = pp.Inv(self.odom_depth_pp[idx]) @ self.odom_depth_pp
            goals = goals.tensor()[goal_frames]

            # filter goals depending on their distance 
            goal_filter = np.linalg.norm(self.odom_depth[goal_frames, :2] - self.odom_depth[idx, :2], axis=1) > self.args.tolerance
            goals = goals[goal_filter]
            if len(goals) == 0:
                continue
            goals = goals.to("cuda")

            # transform input
            depth_img = transform(depth_img).unsqueeze(0).repeat(len(goals), 1, 1, 1)
            depth_img = depth_img.to("cuda")
            
            # get prediction
            if train_config.sem or train_config.rgb:
                bgr_sem_img_warp = transform(bgr_sem_img_warp).unsqueeze(0).repeat(len(goals), 1, 1, 1).to(torch.float32) / 255.0
                bgr_sem_img_warp = bgr_sem_img_warp.to("cuda")
                with torch.no_grad():
                    preds, fear = trainer.net(depth_img, bgr_sem_img_warp, goals)
            else:
                with torch.no_grad():
                    preds, fear = trainer.net(depth_img, goals)
            
            # optimize
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=0.1)
            waypoints_world = TrajCost.TransformPoints(self.odom_depth[idx][None, :], waypoints.to("cpu")).tensor().cpu().numpy()
            
            # evaluate
            self.goal_distances[pred_counter:pred_counter+len(goals)] = np.linalg.norm(waypoints_world[:, -1, :2] - self.odom_depth[goal_frames[goal_filter], :2], axis=1)
            self.length_path[pred_counter:pred_counter+len(goals)]    = np.sum(np.linalg.norm(waypoints_world[:, 1:, :2] - waypoints_world[:, :-1, :2], axis=2), axis=1)
            self.length_goal[pred_counter:pred_counter+len(goals)]    = np.linalg.norm(self.odom_depth[goal_frames[goal_filter], :2] - self.odom_depth[idx, :2], axis=1)
            pred_counter += len(goals)

            # viz trajectories within the depth image
            if self.args.save_viz:
                cv_img_list = traj_viz.VizImages(
                    preds=preds,
                    waypoints=waypoints,
                    odom=torch.tensor(self.odom_depth[idx][None, :], device=waypoints.device, dtype=torch.float32),
                    goal=goals,
                    fear=fear,
                    images=depth_img,
                    is_shown=False,
                )
                raise NotImplementedError("Before Progressing, check if bgr or rgb input")
                [cv2.imwrite(os.path.join(self.args.data_dir, "viz", f"{idx}_{i}.png"), cv_img) for i, cv_img in enumerate(cv_img_list)]
            
        # crop buffers
        self.goal_distances = self.goal_distances[:pred_counter]
        self.length_path = self.length_path[:pred_counter]
        self.length_goal = self.length_goal[:pred_counter]
        
        # sort values
        sort_indices = np.argsort(self.length_goal)
        self.length_goal = self.length_goal[sort_indices]
        self.length_path = self.length_path[sort_indices]
        self.goal_distances = self.goal_distances[sort_indices]

        # make directory and save data
        _, model_name = os.path.split(model_dir)
        eval_dir = os.path.join(self.args.data_dir, f"eval_{model_name}")
        os.makedirs(eval_dir, exist_ok=True)

        np.savetxt(os.path.join(eval_dir, "length_path.txt"), self.length_path)
        np.savetxt(os.path.join(eval_dir, "length_goal.txt"), self.length_goal)
        np.savetxt(os.path.join(eval_dir, "goal_distances.txt"), self.goal_distances)

        # plot data
        self.plt_single_model(eval_dir)
        
        # get statistics
        self.eval_statistics()
        self.save_eval_results(model_dir, save_name=os.path.split(self.args.data_dir)[-1])
        
        return self.length_goal, self.length_path, self.goal_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Model Eval', description='Evaluate VIPmodels with real-world data')
    parser.add_argument('-m', '--model_dirs', nargs='+', type=str, help='Path to model directory',
                        default=[
                            "/home/pascal/SemNav/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD",
                            "/home/pascal/SemNav/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDep_costSem_optimSGD_depth",
                            # "/home/pascal/SemNav/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD_decoderS",
                            # "/home/pascal/SemNav/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD_obs03_sdecoder",
                            # "/home/pascal/SemNav/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD_obs04_sdecoder",
                        ])
    parser.add_argument('-d', '--data_dir',  type=str, help='Path to data directory (should contain bgr, depth and odom data)', 
                        default="/home/pascal/SemNav/env/anymal/2023_03_23_rsl")
    parser.add_argument('--tolerance', type=float, help='Tolerance to the goal to be considered reached',
                        default=0.5)
    parser.add_argument('--goal_frame_advances', nargs='+', type=int, help='Number of frames to advance the goal',
                        default=[20, 50, 100, 200])
    parser.add_argument('--save_viz', action='store_true', help='Save visualizations of the predictions')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()
    
    m2f_cfg = Mask2FormerCfg()
    RealWorldEvaluator(args, m2f_cfg)

# EoF
 