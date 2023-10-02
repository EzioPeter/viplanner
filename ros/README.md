# ViPlanner ROS Node

## Overview

ROS Node to run VIPlanner on the LeggedRobot Platform ANYmal.
The implementation consists of
- the planner itself, running a semantic segmentation network and VIPlanner in parallel
- a visualization node to project the path in the RGB and depth camera stream of the ANYmal
- a pathFollower to translate the path into twist commands that can be executed by the robot
- an RViz plugin to set the waypoints for the planner


## Installation

Please refer to [Installation Instructions](./INSTALL.md) where details about the included docker and a manual install is given.

## Usage

Run the VIPlanner with visualization:

	roslaunch viplanner_node viplanner.launch

## SmartJoystick

Press the **LB** button on the joystick, when seeing the output on the screen:

    Switch to Smart Joystick mode ...

Now the smartjoystick feature is enabled. It takes the joy stick command as motion intention and runs the VIPlanner in the background for low-level obstacle avoidance.
