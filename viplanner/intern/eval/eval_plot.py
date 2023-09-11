"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      Project real-world predicted path in semantic and depth image
"""

# python
import os

import cv2
import numpy as np
import scipy.spatial.transform as tf
import torch

from viplanner.intern.rosbag.rosbag_base_handler import (
    CAMERA_FLIP_MAT,
    ROS_TO_ROBOTICS_MAT,
    RealWorldDataHandler,
)

# viplanner
from viplanner.traj_cost_opt import TrajViz


class EvalPlotter(RealWorldDataHandler):
    def __init__(
        self,
        dir: str,
        rotate: bool = False,
    ) -> None:
        super().__init__(dir, rotate)

        self.traj_viz_depth = TrajViz(
            intrinsics=self.K_depth,
            cam_resolution=self.depth_img_shape,
        )
        self.traj_viz_rgb = TrajViz(
            intrinsics=self.K_bgr,
            cam_resolution=self.bgr_img_shape,
        )
        return

    def load_data(self):
        super().load_data()
        # get path
        path = np.load(os.path.join(self.dir, "path.npy"))  # 3D
        path_idx, img_idx = self.synchronize_data(
            self.depth_time, path[:, 0, 3:], threshold=0.2
        )

        # path given in camera convention, transform to robot convention
        path_depth = (
            path[path_idx, :, :3] - self.odom_depth[img_idx, None, :3]
        ) @ tf.Rotation.from_quat(self.odom_depth[img_idx, 3:]).as_matrix()
        path_rgb = (
            path[path_idx, :, :3] - self.odom_bgr[img_idx, None, :3]
        ) @ tf.Rotation.from_quat(self.odom_bgr[img_idx, 3:]).as_matrix()

        # reduce data to valid depth images, bgr and odom points
        self.odom_bgr = torch.from_numpy(self.odom_bgr[img_idx]).float()
        self.odom_depth = torch.from_numpy(self.odom_depth[img_idx]).float()
        self.path_depth = torch.from_numpy(path_depth).float()
        self.path_rgb = torch.from_numpy(path_rgb).float()
        self.odom_depth_pp = self.odom_depth_pp[img_idx]
        self.depth_time = self.depth_time[img_idx]
        self.depth_img_list = self.depth_img_list[img_idx]
        self.bgr_img_list = self.bgr_img_list[img_idx]

        # get goal
        goals = np.loadtxt(os.path.join(self.dir, "odom_goal.txt"))
        if len(goals.shape) == 1:
            goals = np.expand_dims(goals, axis=0)
        goal_switch_idx = self.synchronize_data(
            self.depth_time, goals[:, 3:], threshold=10.0
        )
        self.goal_depth = np.zeros((len(self.path_depth), 3))
        self.goal_rgb = np.zeros((len(self.path_rgb), 3))
        goal_idx = None
        remove_idx = []
        for idx in range(len(self.path_depth)):
            if (idx == goal_switch_idx[1]).any():
                goal_idx = goal_switch_idx[0][
                    np.where(idx == goal_switch_idx[1])[0][0]
                ]
            elif goal_idx is None:
                remove_idx.append(idx)
                continue
            self.goal_depth[idx] = (
                goals[goal_idx, :3] - self.odom_depth[idx, :3].numpy()
            ) @ tf.Rotation.from_quat(
                self.odom_depth[idx, 3:].numpy()
            ).as_matrix()
            self.goal_rgb[idx] = (
                goals[goal_idx, :3] - self.odom_bgr[idx, :3].numpy()
            ) @ tf.Rotation.from_quat(
                self.odom_bgr[idx, 3:].numpy()
            ).as_matrix()

        if len(remove_idx) > 0:
            remove_bool = np.ones(len(self.goal_depth), dtype=bool)
            remove_bool[remove_idx] = False

            self.goal_depth = self.goal_depth[remove_bool]
            self.goal_rgb = self.goal_rgb[remove_bool]
            self.depth_time = self.depth_time[remove_bool]
            self.depth_img_list = self.depth_img_list[remove_bool]
            self.bgr_img_list = self.bgr_img_list[remove_bool]
            self.odom_bgr = self.odom_bgr[remove_bool]
            self.odom_depth = self.odom_depth[remove_bool]
            self.path_depth = self.path_depth[remove_bool]
            self.path_rgb = self.path_rgb[remove_bool]
            self.odom_depth_pp = self.odom_depth_pp[remove_bool]

        self.goal_depth = torch.tensor(self.goal_depth)
        self.goal_rgb = torch.tensor(self.goal_rgb)

        # get depth and rgb image dimensions
        self.depth_img_shape = np.shape(
            cv2.imread(
                os.path.join(self.dir, "depth", self.depth_img_list[0]),
                cv2.IMREAD_UNCHANGED,
            )
        )
        self.bgr_img_shape = np.shape(
            cv2.imread(os.path.join(self.dir, "bgr", self.bgr_img_list[0]))
        )

        # semantics
        if self.semantics:
            self.sem_img_list = self.sem_img_list[img_idx]
            self.sem_img = torch.zeros(
                (
                    len(self.sem_img_list),
                    self.bgr_img_shape[2],
                    self.bgr_img_shape[0],
                    self.bgr_img_shape[1],
                )
            )

        # load all images
        self.depth_img = torch.zeros(
            (
                len(self.depth_img_list),
                self.depth_img_shape[0],
                self.depth_img_shape[1],
            )
        )
        self.bgr_img = torch.zeros(
            (
                len(self.bgr_img_list),
                self.bgr_img_shape[2],
                self.bgr_img_shape[0],
                self.bgr_img_shape[1],
            )
        )
        for idx in range(len(self.depth_img_list)):
            self.depth_img[idx] = torch.from_numpy(
                cv2.imread(
                    os.path.join(self.dir, "depth", self.depth_img_list[idx]),
                    cv2.IMREAD_UNCHANGED,
                )
                / 1000.0
            )  # rotate by 180 deg
            bgr_img = cv2.imread(
                os.path.join(self.dir, "bgr", self.bgr_img_list[idx])
            )
            self.bgr_img[idx] = (
                torch.from_numpy(
                    cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                ).permute(2, 0, 1)
                / 255.0
            )

            if self.semantics:
                sem_img = cv2.imread(
                    os.path.join(self.dir, "semantics", self.sem_img_list[idx])
                )
                self.sem_img[idx] = (
                    torch.from_numpy(
                        cv2.cvtColor(sem_img, cv2.COLOR_BGR2RGB)
                    ).permute(2, 0, 1)
                    / 255.0
                )

        return

    def plot(self):
        projected_img_depth = self.traj_viz_depth.VizImages(
            preds=self.path_depth[:, [0, 9, 19, 29, 39, 49]],
            waypoints=self.path_depth,
            odom=self.odom_depth,
            goal=self.goal_depth,
            fear=np.zeros((len(self.path_depth), 1)),
            images=self.depth_img,
            is_shown=False,
        )

        projected_img_rgb = self.traj_viz_rgb.VizImages(
            preds=self.path_rgb[:, [0, 9, 19, 29, 39, 49]],
            waypoints=self.path_rgb,
            odom=self.odom_bgr,
            goal=self.goal_rgb,
            fear=np.zeros((len(self.path_rgb), 1)),
            images=self.bgr_img,
            is_shown=False,
        )

        if self.semantics:
            projected_img_sem = self.traj_viz_rgb.VizImages(
                preds=self.path_rgb[:, [0, 9, 19, 29, 39, 49]],
                waypoints=self.path_rgb,
                odom=self.odom_bgr,
                goal=self.goal_rgb,
                fear=np.zeros((len(self.path_rgb), 1)),
                images=self.sem_img,
                is_shown=False,
            )

        # save images
        os.makedirs(os.path.join(self.dir, "depth_projected"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "rgb_projected"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "sem_projected"), exist_ok=True)

        for idx in range(len(projected_img_depth)):
            cv2.imwrite(
                os.path.join(
                    self.dir, "depth_projected", self.depth_img_list[idx]
                ),
                projected_img_depth[idx],
            )
            cv2.imwrite(
                os.path.join(
                    self.dir, "rgb_projected", self.bgr_img_list[idx]
                ),
                projected_img_rgb[idx],
            )
            if self.semantics:
                assert cv2.imwrite(
                    os.path.join(
                        self.dir, "sem_projected", self.bgr_img_list[idx]
                    ),
                    projected_img_sem[idx],
                )


if __name__ == "__main__":
    dir = "/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_stairs_both_door"
    eval_plotter = EvalPlotter(dir, rotate=True)
    eval_plotter.plot()

# EoF
