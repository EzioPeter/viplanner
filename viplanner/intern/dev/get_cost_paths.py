# python
import os

import numpy as np
import torch

# imperative-planning-learning
from viplanner.traj_cost_opt import TrajCost

# viplanner sem
# model = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse"

# viplanner geom
# model = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDep_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse_depth"

# iplanner
model = "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models"

datasets = [
    "eval_2n8kARJN3HM_new_pathfollower",
    "eval_Town01_Opt_large_scale",
    "eval_warehouse_multiple_shelves_without_ppl_ext_sem_space",
]

prefixs = [
    "2n8kARJN3HM_seed1_pairs500",
    "waypoints_carla_eval",
    "warehouse_multiple_shelves_without_ppl_ext_sem_space_seed1_pairs500",
]

costmap_dirs = [
    "/home/pascal/viplanner/imperative_learning/data/2n8kARJN3HM_cam_mount",
    "/home/pascal/viplanner/imperative_learning/data/town01_cam_mount_train",
    "/home/pascal/viplanner/imperative_learning/data/warehouse_multiple_shelves_without_ppl_ext_sem_space",
]

import pickle

with open("/home/pascal/viplanner/imperative_learning/data/waypoints/2n8kARJN3HM_seed1_pairs500.pkl", "rb") as f:
    waypoints = pickle.load(f)

goals = []
for curr_waypoint in waypoints:
    goals.append(np.array(list(curr_waypoint.values())))
goals = np.vstack(goals)

goals_allowed = np.where((goals[:, 0] > -9.0) & (goals[:, 0] < 8.3))[0]
bool_goal = np.zeros(len(goals), dtype=bool)
bool_goal[goals_allowed] = True

for idx, curr_dataset in enumerate(datasets):
    goal_reached = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_goal_reached.npy"))
    goal_within_fov = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_goal_within_fov.npy"))
    base_collision = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_base_collision.npy"))
    knee_collision = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_knee_collision.npy"))
    walking_time = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_walking_time.npy"))
    goal_distances = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_goal_distances.npy"))
    length_goal = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_length_goal.npy"))
    length_path = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_length_path.npy"))
    loss_obstacles_max_sem = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_loss_obstacles.npy"))
    skip_waypoint = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_skip_waypoint.npy"))
    path_extension = np.load(os.path.join(model, curr_dataset, prefixs[idx] + "_path_extension.npy"))

    # get geometric max loss along the path
    traj_cost = TrajCost()
    traj_cost.SetMap(costmap_dirs[idx], "tsdf_sem")
    loss_obstacles_max_geom = np.zeros_like(loss_obstacles_max_sem)

    filter_idx = []

    for waypoint_nbr in range(len(loss_obstacles_max_sem)):
        if "warehouse" in costmap_dirs[idx] and not bool_goal[waypoint_nbr]:
            filter_idx.append(waypoint_nbr)
            continue

        if not goal_reached[waypoint_nbr]:
            loss_obstacles_max_geom[waypoint_nbr] = np.nan
            # if "warehouse" in costmap_dirs[idx]:
            #     filter_idx.append(waypoint_nbr)
            continue

        # load waypoints
        curr_waypoint = np.load(os.path.join(model, curr_dataset, f"waypoint{waypoint_nbr}_path.npy"))

        if "warehouse" in costmap_dirs[idx]:
            goes_through_shelf = np.where(
                (curr_waypoint[:, 0] < 0.2)
                & (curr_waypoint[:, 0] > -0.9)
                & (curr_waypoint[:, 1] > -3.6)
                & (curr_waypoint[:, 1] < 12.56)
            )[0]
            if len(goes_through_shelf) > 0:
                filter_idx.append(waypoint_nbr)
                continue

        # get its geoemtric loss
        waypoints = torch.tensor(curr_waypoint, dtype=torch.float32).cuda()
        loss_obstacles_max_geom[waypoint_nbr] = traj_cost.cost_of_recorded_path(waypoints).cpu().numpy()

    if "warehouse" in costmap_dirs[idx]:
        filter_idx = [
            32,
            44,
            48,
            62,
            70,
            77,
            90,
            94,
            117,
            124,
            125,
            141,
            143,
            146,
            149,
            165,
            167,
            174,
            176,
            187,
            197,
            202,
            206,
            216,
            218,
            225,
            228,
            233,
            238,
            240,
            241,
            244,
            248,
            249,
            258,
            261,
            262,
            267,
            273,
            278,
            290,
            294,
            296,
            298,
            305,
            307,
            315,
            318,
            320,
            327,
            328,
            333,
            334,
            336,
            339,
            340,
            346,
            348,
            350,
            352,
            354,
            357,
            361,
            363,
            364,
            366,
            367,
            368,
            370,
            371,
            375,
            376,
            377,
            379,
            380,
            383,
            385,
            387,
            392,
            393,
            395,
            398,
            399,
            400,
            402,
            403,
            405,
            406,
            416,
            417,
            423,
            425,
            427,
            432,
            433,
            438,
            445,
            448,
            450,
            451,
            453,
            454,
            457,
            465,
            467,
            470,
            472,
            473,
            475,
            476,
            483,
            485,
            487,
            488,
            490,
            32,
            33,
            335,
            336,
            353,
            354,
            427,
            446,
            452,
            469,
            471,
            298,
            301,
            296,
            482,
            489,
            391,
            397,
            398,
            213,
            215,
            218,
        ]
        bool_array = np.ones_like(loss_obstacles_max_sem, dtype=bool)
        bool_array[filter_idx] = False
        # filter all
        loss_obstacles_max_sem = loss_obstacles_max_sem[bool_array]
        loss_obstacles_max_geom = loss_obstacles_max_geom[bool_array]
        goal_reached = goal_reached[bool_array]
        path_extension = path_extension[bool_array]
        skip_waypoint = skip_waypoint[bool_array]

    # get statistics
    sem_mean = np.mean(loss_obstacles_max_sem[goal_reached])
    sem_std = np.std(loss_obstacles_max_sem[goal_reached])
    geom_mean = np.mean(loss_obstacles_max_geom[goal_reached])
    geom_std = np.std(loss_obstacles_max_geom[goal_reached])
    path_extension_mean = np.mean(path_extension[goal_reached])
    path_extension_std = np.std(path_extension[goal_reached])

    print(f"Dataset: {curr_dataset}")
    print(f"Success rate: {np.sum(goal_reached[~skip_waypoint]) / len(goal_reached[~skip_waypoint]) * 100}")
    print(f"Geometric mean: {geom_mean:.3f} +- {geom_std:.3f}")
    print(f"Semantic mean: {sem_mean:.3f} +- {sem_std:.3f}")
    print(f"Path extension mean: {path_extension_mean:.3f} +- {path_extension_std:.3f}")
    print("-----------------------------------------------")
