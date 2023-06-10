#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      Base Function to handle the data extracted from rosbags
"""

# python
import os
import numpy as np
import pypose as pp
import scipy.spatial.transform as tf

ROS_TO_ROBOTICS_MAT = tf.Rotation.from_euler("XYZ", [-90, 0, -90], degrees=True).as_matrix()
CAMERA_FLIP_MAT     = tf.Rotation.from_euler("XYZ", [180, 0, 0],   degrees=True).as_matrix()
time_threshold = 0.1


class RealWorldDataHandler:

    def __init__(self, dir: str, rotate: bool = False) -> None:
        self.dir = dir
        self.rotate = rotate  # true for ANYmal D and False for ANYmal C
        
        if os.path.exists(os.path.join(self.dir, "semantics")):
            self.semantics = True
        else:
            self.semantics = False

        # load data
        self.K_bgr: np.ndarray = None
        self.K_depth: np.ndarray = None
        self.odom_bgr: np.ndarray = None
        self.odom_depth: np.ndarray = None
        self.depth_img_list: np.ndarray = None
        self.bgr_img_list: np.ndarray = None
        self.sem_img_list: np.ndarray = None
        self.odom_depth_pp: pp.LieTensor = None
        self.load_data()
        
        return

    def load_data(self) -> None:
        # load data timestamps and synchroniize them
        odom_bgr   = np.loadtxt(os.path.join(self.dir, "odom_bgr.txt"))
        odom_depth = np.loadtxt(os.path.join(self.dir, "odom_depth.txt"))
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
        self.K_depth = np.loadtxt(os.path.join(self.dir, "intrinsics_depth.txt"))
        self.K_bgr   = np.loadtxt(os.path.join(self.dir, "intrinsics_bgr.txt"))
        
        # reduce data to valid depth images, bgr and odom points
        self.depth_img_list = np.array(sorted(os.listdir(os.path.join(self.dir, "depth"))))
        self.depth_img_list = self.depth_img_list[depth_idx] 
        self.bgr_img_list   = np.array(sorted(os.listdir(os.path.join(self.dir, "bgr"))))
        self.bgr_img_list   = self.bgr_img_list[bgr_idx]
        self.depth_time = odom_depth[depth_idx, 7:]
        odom_depth     = odom_depth[depth_idx, :7]
        odom_bgr       = odom_bgr[bgr_idx, :7]

        if self.semantics:
            self.sem_img_list   = np.array(sorted(os.listdir(os.path.join(self.dir, "semantics"))))
            self.sem_img_list   = self.sem_img_list[bgr_idx]
        
        # transform rotations of depth and bgr image from ROS camera frame (z-forward) to robotics frame (x-forward)
        depth_rot_mat = tf.Rotation.from_quat(odom_depth[:, 3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT
        if not self.rotate:
            depth_rot_mat = depth_rot_mat @ CAMERA_FLIP_MAT
        odom_depth[:, 3:] = tf.Rotation.from_matrix(depth_rot_mat).as_quat()
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