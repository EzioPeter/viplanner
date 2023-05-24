# Visual Imperative Planner (VIPlanner)

## Overview

Imperative learning based visual local planner using front depth and RGB image

**Keywords:** Visual Navigation, Local Planning, Imperative Learning

### License

This code belongs to Robotic Systems Lab, ETH Zurich. 
All right reserved

**Author: Pascal Roth<br />
Maintainer: Pascal Roth, rothpa@ethz.ch**

The VIPlanner package has been tested under ROS Noetic on Ubuntu 20.04.
This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.


## Installation

Please refer to [Installation Instructions](INSTALL.md)

## Usage

Run the VIPlanner without visualization:

	roslaunch viplanner_node viplanner.launch 

## SmartJoystick

Press the **LB** button on the joystick, when seeing the output on the screen:

    Switch to Smart Joystick mode ...

Now the smartjoystick feature is enabled. It takes the joy stick command as motion intension and runs the VIPlanner in the background for low-level obstacle avoidance.
