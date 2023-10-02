import numpy as np
import os

threshold = 0.5

# viplanner
# model_dir = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse"
# iplanner
model_dir = "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models"

# environments
eval_dir = [
    "eval_Town01_Opt_large_scale",
    "eval_warehouse_multiple_shelves_without_ppl_ext_sem_space",
    "eval_2n8kARJN3HM_new_pathfollower",
]
prefix = [
    "waypoints_carla_eval",
    "warehouse_multiple_shelves_without_ppl_ext_sem_space_seed1_pairs500",
    "2n8kARJN3HM_seed1_pairs500",
]


for idx, curr_eval_dir in enumerate(eval_dir):
    curr_prefix = prefix[idx]

    goal_reached = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_goal_reached.npy"))
    goal_within_fov = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_goal_within_fov.npy"))
    base_collision = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_base_collision.npy"))
    knee_collision = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_knee_collision.npy"))
    walking_time = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_walking_time.npy"))
    goal_distances = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_goal_distances.npy"))
    length_goal = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_length_goal.npy"))
    length_path = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_length_path.npy"))
    loss_obstacles = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_loss_obstacles.npy"))
    skip_waypoint = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_skip_waypoint.npy"))
    path_extension = np.load(os.path.join(model_dir, curr_eval_dir, curr_prefix + f"_path_extension.npy"))

    # filter skip waypoints
    goal_reached = goal_reached[~skip_waypoint]
    loss_obstacles = loss_obstacles[~skip_waypoint] 

    # recorded is max loss along the path!!!!
    # goal reached and loss under 1.0
    success = np.logical_and(goal_reached, loss_obstacles < threshold)
    model = os.path.split(curr_eval_dir)[-1]
    print(f"{model}: Threshold: {threshold} Success rate: {np.sum(success) / len(success)}")
