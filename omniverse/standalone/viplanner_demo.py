# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-orbit
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--conv_distance", default=0.2, type=float, help="Distance for a goal considered to be reached.")
parser.add_argument("--scene", default="matterport", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from omni.isaac.orbit.envs import NavigationEnv
from omni.viplanner.config import ViPlannerMatterportCfg

# viplanner
from omni.viplanner.viplanner import VIPlannerAlgo

"""
Main
"""


def main():
    """Imports all legged robots supported in Orbit and applies zero actions."""

    # create environment
    env_cfg = ViPlannerMatterportCfg()
    env = NavigationEnv(env_cfg)

    # load viplanner
    viplanner = VIPlannerAlgo(
        model_dir="/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse"
    )

    goals = torch.tensor([[5.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)

    # initial paths
    paths = torch.zeros((env.num_envs, 50, 3), device=env.device)

    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if env.sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not env.sim.is_playing():
            env.sim.step(render=app_launcher.RENDER)
            continue

        planner_obs, _ = env.step(paths=paths)
        env.render()

        # apply planner
        goal_cam_frame = viplanner.goal_transformer(goals, planner_obs["cam_position"], planner_obs["cam_orientation"])
        _, paths, fear = viplanner.plan_dual(
            planner_obs["depth_measurement"], planner_obs["semantic_measurement"], goal_cam_frame
        )
        paths = viplanner.path_transformer(paths, planner_obs["cam_position"], planner_obs["cam_orientation"])

        # draw path
        viplanner.debug_draw(paths, fear, goals, planner_obs["cam_position"])


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
