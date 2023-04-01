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
from config import TrainCfg, VIPlannerSemMetaHandler, get_class_for_id
from utils.trainer import Trainer
from utils.m2f_utils import load_m2f_demo
from traj_cost_opt import TrajOpt, TrajCost, TrajViz
from dataset import PlannerDataGenerator

# set random seed for reproducibility
torch.manual_seed(12)
time_threshold = 0.1

ROS_TO_ROBOTICS_MAT = tf.Rotation.from_euler("XYZ", [-90, 0, -90], degrees=True).as_matrix()
CAMERA_FLIP_MAT     = tf.Rotation.from_euler("XYZ", [180, 0, 0],   degrees=True).as_matrix()


def sem_predictions(args: argparse.Namespace, bgr_list: List[str], debug: bool = False) -> None:
    m2f_demo = load_m2f_demo(args.m2f_weights, args.m2f_cfg)

    # get mapping from coco to viplanner semantic classes
    viplanner_meta = VIPlannerSemMetaHandler()
    coco_viplanner_cls_mapping = get_class_for_id()
    viplanner_sem_class_color_map = viplanner_meta.class_color
    coco_viplanner_color_mapping = {}
    for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
        coco_viplanner_color_mapping[coco_id] = viplanner_meta.class_color[viplanner_cls_name]
    
    # make new diretory for semantic classes
    os.makedirs(os.path.join(args.data_dir, "semantics"), exist_ok=True)

    # check if and which files are already in the directory, only progress for the non-included files
    files_in_directory_set = set(os.listdir(os.path.join(args.data_dir, "semantics")))
    filenames_set = set(bgr_list)
    filenames_not_included = list(filenames_set - files_in_directory_set)
    if len(filenames_not_included) == 0:
        return
    
    # get list of all files in bgr directory
    for curr_bgr in tqdm(filenames_not_included, desc="Semantic Estimation"):
        # load_image
        image = cv2.imread(os.path.join(args.data_dir, "bgr", curr_bgr))
        # get predictions
        predictions, visualized_output = m2f_demo.run_on_image(image)
        panoptic_seg, seg_infos = predictions['panoptic_seg']
        # create output
        panoptic_mask = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1], 3), dtype=np.uint8)
        for sinfo in seg_infos:
            try:
                panoptic_mask[panoptic_seg.cpu().numpy() == sinfo['id']] = coco_viplanner_color_mapping[sinfo['category_id']]
            except KeyError:
                print(f"WARNING: Category {sinfo['category_id']} not found in coco_viplanner_cls_mapping.")
                panoptic_mask[panoptic_seg.cpu().numpy() == sinfo['id']] = viplanner_sem_class_color_map['static']
        
        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(panoptic_mask)
            plt.show()
               
        # save image 
        panoptic_mask = cv2.cvtColor(panoptic_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.data_dir, "semantics", curr_bgr), panoptic_mask)
    return


def synchronize_data(bgr_timestamp, depth_timestamp, threshold=0.1):
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
    
    
def main(args) -> None:
    """
    Make prediction on real world images and evaluate the generated pathes. Expected args:

    args.model_dir: str     -- model dir
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
    args.m2f_cfg            -- path to mask2former config yaml file
    args.m2f_weights        -- path to mask2former weight .pkl file
    """

    # load config
    train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(args.model_dir, "model.yaml"))
    
    # load data timestamps and synchroniize them
    odom_bgr   = np.loadtxt(os.path.join(args.data_dir, "odom_bgr.txt"))
    odom_depth = np.loadtxt(os.path.join(args.data_dir, "odom_depth.txt"))
    depth_idx, bgr_idx = synchronize_data(odom_bgr[:, 7:], odom_depth[:, 7:])
    
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
    K_depth = np.loadtxt(os.path.join(args.data_dir, "intrinsics_depth.txt"))
    K_bgr   = np.loadtxt(os.path.join(args.data_dir, "intrinsics_bgr.txt"))
    
    # reduce data to valid depth images, bgr and odom points
    depth_img_list = np.array(sorted(os.listdir(os.path.join(args.data_dir, "depth"))))
    depth_img_list = depth_img_list[depth_idx] 
    bgr_img_list   = np.array(sorted(os.listdir(os.path.join(args.data_dir, "bgr"))))
    bgr_img_list   = bgr_img_list[bgr_idx]
    odom_depth     = odom_depth[depth_idx, :7]
    odom_bgr       = odom_bgr[bgr_idx, :7]
    
    # transform rotations of depth and bgr image from ROS camera frame (z-forward) to robotics frame (x-forward)
    odom_depth[:, 3:] = tf.Rotation.from_matrix(tf.Rotation.from_quat(odom_depth[:, 3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT).as_quat()
    odom_bgr[:, 3:] = tf.Rotation.from_matrix(tf.Rotation.from_quat(odom_bgr[:, 3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT @ CAMERA_FLIP_MAT).as_quat()
    odom_depth_pp = pp.SE3(odom_depth)
    
    # if semantics are used, load M2F model and estimate semantics
    if train_config.sem:
        sem_predictions(args, bgr_list=bgr_img_list, debug=args.debug)

    # load trainer, model
    trainer = Trainer(train_config)
    trainer._load_model(resume=True)

    # image transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((train_config.img_input_size), antialias=True),
    ])

    # make directory for visualization results
    if args.save_viz:
        os.makedirs(os.path.join(args.data_dir, "viz"), exist_ok=True)
        
    # init trajectory optimizer and visualizer
    traj_opt = TrajOpt()
    traj_viz = TrajViz(intrinsics=K_depth, cam_resolution=train_config.img_input_size)
    
    # init pixel array
    depth_img = cv2.imread(os.path.join(args.data_dir, "depth", depth_img_list[0]), cv2.IMREAD_ANYDEPTH)
    depth_img = np.asarray(depth_img)
    x_nums, y_nums = depth_img.shape
    depth_pixel_array = PlannerDataGenerator.compute_pixel_tensor(x_nums, y_nums, K_depth)

    # init eval buffers
    length_path    = np.zeros((len(depth_img_list) - max(args.goal_frame_advances)) * len(args.goal_frame_advances))
    length_goal    = np.zeros((len(depth_img_list) - max(args.goal_frame_advances)) * len(args.goal_frame_advances))
    goal_distances = np.zeros((len(depth_img_list) - max(args.goal_frame_advances)) * len(args.goal_frame_advances))

    # make predictions
    pred_counter = 0
    for idx, img in enumerate(tqdm(depth_img_list[:-max(args.goal_frame_advances)], desc="VIPlanner Predictions")):
        # load images
        depth_img = cv2.imread(os.path.join(args.data_dir, "depth", img), cv2.IMREAD_ANYDEPTH)
        depth_img = depth_img.astype(np.float32) / 1000.0
        depth_img[depth_img > train_config.data_cfg.max_depth] = 0.0
        depth_img[~np.isfinite(depth_img)] = 0.0
        if train_config.sem:
            bgr_sem_img = cv2.imread(os.path.join(args.data_dir, "semantics", bgr_img_list[idx]))
        else:
            bgr_sem_img = cv2.imread(os.path.join(args.data_dir, "bgr", bgr_img_list[idx]))
            raise NotImplementedError("Before Progressing, check if bgr or rgb input")
        bgr_sem_img = cv2.cvtColor(bgr_sem_img, cv2.COLOR_BGR2RGB)  # since loaded as BGR with cv2

        # warp rgb/sem image onto depth image
        bgr_sem_img_warp = PlannerDataGenerator.compute_overlay(odom_depth[idx], odom_bgr[idx], depth_img=depth_img, sem_rgb_image=bgr_sem_img, pix_depth_cam_frame=depth_pixel_array, K_sem_rgb=K_bgr)
        
        # DEBUG
        if args.debug:
            import matplotlib.pyplot as plt
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(depth_img)
            ax2.imshow(bgr_sem_img)
            ax3.imshow(depth_img)
            ax3.imshow(bgr_sem_img_warp / 255, alpha=0.5)
            plt.show()
            
        # select goal
        goal_frames = np.array(args.goal_frame_advances) + idx
        goals = pp.Inv(odom_depth_pp[idx]) @ odom_depth_pp
        goals = goals.tensor()[goal_frames]

        # filter goals depending on their distance 
        goal_filter = np.linalg.norm(odom_depth[goal_frames, :2] - odom_depth[idx, :2], axis=1) > args.tolerance
        goals = goals[goal_filter]
        if len(goals) == 0:
            continue
        
        # transform input
        depth_img = transform(depth_img).unsqueeze(0).repeat(len(goals), 1, 1, 1)
        bgr_sem_img_warp = transform(bgr_sem_img_warp).unsqueeze(0).repeat(len(goals), 1, 1, 1).to(torch.float32) / 255.0
        
        # transform to model device
        depth_img = depth_img.to("cuda")
        bgr_sem_img_warp = bgr_sem_img_warp.to("cuda")
        goals = goals.to("cuda")
        
        # get prediction and optimize 
        with torch.no_grad():
            preds, fear = trainer.net(depth_img, bgr_sem_img_warp, goals)
            waypoints = traj_opt.TrajGeneratorFromPFreeRot(preds, step=0.1)
            waypoints_world = TrajCost.TransformPoints(odom_depth[idx][None, :], waypoints.to("cpu")).tensor().cpu().numpy()
        
        # evaluate
        goal_distances[pred_counter:pred_counter+len(goals)] = np.linalg.norm(waypoints_world[:, -1, :2] - odom_depth[goal_frames[goal_filter], :2], axis=1)
        length_path[pred_counter:pred_counter+len(goals)]    = np.sum(np.linalg.norm(waypoints_world[:, 1:, :2] - waypoints_world[:, :-1, :2], axis=2), axis=1)
        length_goal[pred_counter:pred_counter+len(goals)]    = np.linalg.norm(odom_depth[goal_frames[goal_filter], :2] - odom_depth[idx, :2], axis=1)
        pred_counter += len(goals)

        # viz trajectories within the depth image
        if args.save_viz:
            cv_img_list = traj_viz.VizImages(
                preds=preds,
                waypoints=waypoints,
                odom=torch.tensor(odom_depth[idx][None, :], device=waypoints.device, dtype=torch.float32),
                goal=goals,
                fear=fear,
                images=depth_img,
                is_shown=False,
            )
            [cv2.imwrite(os.path.join(args.data_dir, "viz", f"{idx}_{i}.png"), cv_img) for i, cv_img in enumerate(cv_img_list)]
        
    # crop buffers
    goal_distances = goal_distances[:pred_counter]
    length_path = length_path[:pred_counter]
    length_goal = length_goal[:pred_counter]
    
    # sort values
    sort_indices = np.argsort(length_goal)
    length_goal = length_goal[sort_indices]
    length_path = length_path[sort_indices]
    goal_distances = goal_distances[sort_indices]

    # make directory and save data
    _, model_name = os.path.split(args.model_dir)
    eval_dir = os.path.join(args.data_dir, f"eval_{model_name}")
    os.makedirs(eval_dir)

    np.savetxt(os.path.join(eval_dir, "length_path.txt"), length_path)
    np.savetxt(os.path.join(eval_dir, "length_goal.txt"), length_goal)
    np.savetxt(os.path.join(eval_dir, "goal_distances.txt"), goal_distances)

    ## plot with the distance to the goal depending on the length between goal and start
    unique_goal_length = np.unique(np.round(length_goal, 1))
    mean_path_length = []
    std_path_length = []
    for x in unique_goal_length:
        y_subset = length_path[np.round(length_goal, 1) == x]
        mean_path_length.append(np.mean(y_subset))
        std_path_length.append(np.std(y_subset))

    avg_increase = np.mean((length_path / length_goal) -1)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle("Path Length Increase", fontsize=20)
    ax.plot(unique_goal_length, mean_path_length, color='blue', label='Average path length')
    ax.fill_between(unique_goal_length, np.array(mean_path_length) - np.array(std_path_length), np.array(mean_path_length) + np.array(std_path_length), color='blue', alpha=0.2, label='Uncertainty')
    ax.set_xlabel('Start-Goal Distance', fontsize=16)
    ax.set_ylabel('Path Length', fontsize=16)
    ax.set_title(f"Avg increase of path length is {round(avg_increase, 5)*100}%", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(eval_dir, "path_length.png"))
    plt.show() 


    ## plot to compare the increase in path length depending in on the distance between goal and start
    mean_goal_distance = []
    std_goal_distance = []
    goal_counts = []

    for x in unique_goal_length:
        y_subset = goal_distances[np.round(length_goal, 1) == x]
        mean_goal_distance.append(np.mean(y_subset))
        std_goal_distance.append(np.std(y_subset))
        goal_counts.append(len(y_subset))

    goal_success = np.sum(goal_distances < args.tolerance) / len(goal_distances)

    # Create a figure and two axis objects, with the second one sharing the x-axis of the first
    fig, ax1 = plt.subplots(figsize=(12, 10))
    ax2 = ax1.twinx()
    fig.subplots_adjust(hspace=0.4)  # Add some vertical spacing between the two plots

    # Plot the goal distance data
    ax1.plot(unique_goal_length, mean_goal_distance, color='blue', label='Average goal distance length', zorder=2)
    ax1.fill_between(unique_goal_length, np.array(mean_goal_distance) - np.array(std_goal_distance), np.array(mean_goal_distance) + np.array(std_goal_distance), color='blue', alpha=0.2, label='Uncertainty', zorder=1)
    ax1.set_xlabel('Start-Goal Distance', fontsize=16)
    ax1.set_ylabel('Goal Distance', fontsize=16)
    ax1.set_title(f"With a tolerance of {args.tolerance} are {round(goal_success, 5)*100} % of goals reached", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # Plot the goal counts data on the second axis
    ax2.bar(unique_goal_length, goal_counts, color='red', alpha=0.5, width=0.05, label='Number of samples', zorder=0)
    ax2.set_ylabel('Sample count', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    # Combine the legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax2.legend(lines+bars, labels+bar_labels, loc='upper center')

    plt.suptitle("Goal Distance", fontsize=20)
    plt.tight_layout()
    fig.savefig(os.path.join(eval_dir, "goal_distance.png"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Model Eval', description='Evaluate VIPmodels with real-world data')
    parser.add_argument('-m', '--model_dir', type=str, help='Path to model directory',
                        default="/home/pascal/SemNav/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD/")
    parser.add_argument('-d', '--data_dir',  type=str, help='Path to data directory (should contain bgr, depth and odom data)', 
                        default="/home/pascal/SemNav/env/anymal/2023_03_23_rsl/")
    parser.add_argument('--m2f_cfg',  type=str, help='Path to config file for mask2former',
                        default="/home/pascal/SemNav/sem_seg/m2f_model/coco/panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml")
    parser.add_argument('--m2f_weights',  type=str, help='Path to weights file for mask2former',
                        default="/home/pascal/SemNav/sem_seg/m2f_model/coco/panoptic/swin/model_final_9fd0ae.pkl")
    parser.add_argument('--tolerance', type=float, help='Tolerance to the goal to be considered reached',
                        default=0.5)
    parser.add_argument('--goal_frame_advances', nargs='+', type=int, help='Number of frames to advance the goal',
                        default=[20, 50, 100, 200])
    parser.add_argument('--save_viz', action='store_true', help='Save visualizations of the predictions')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()
    
    main(args)

# EoF
 