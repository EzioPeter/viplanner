# VIPlanner ROS Node

## Overview

ROS Node to run VIPlanner on the LeggedRobot Platform ANYmal.

## Installation

Please refer to [Installation Instructions](./INSTALL.md)

## Usage

Run the VIPlanner without visualization:

	roslaunch viplanner_node viplanner.launch 

## SmartJoystick

Press the **LB** button on the joystick, when seeing the output on the screen:

    Switch to Smart Joystick mode ...

Now the smartjoystick feature is enabled. It takes the joy stick command as motion intension and runs the VIPlanner in the background for low-level obstacle avoidance.
