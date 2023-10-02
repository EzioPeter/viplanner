#!/usr/bin python3

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

# viplanner
from viplanner.traj_cost_opt import TrajViz
from viplanner.traj_cost_opt.traj_opt import TrajOpt
from viplanner.intern.rosbag.rosbag_base_handler import (
    ROS_TO_ROBOTICS_MAT,
    RealWorldDataHandler,
)


class EvalVideo:
    def __init__(
        self,
        dir: str,
    ) -> None:
        self.dir = dir

        # load intrinsics
        self.K_depth = np.loadtxt(os.path.join(self.dir, "intrinsics_depth.txt"))
        self.K_bgr = np.loadtxt(os.path.join(self.dir, "intrinsics_bgr.txt"))
        self.load_transforms()

        # get depth and rgb image dimensions
        self.depth_img_shape = np.shape(
            cv2.imread(os.path.join(self.dir, "depth", self.depth_img_list[0]), cv2.IMREAD_UNCHANGED)
        )
        self.bgr_img_shape = np.shape(cv2.imread(os.path.join(self.dir, "bgr", self.bgr_img_list[0])))

        # init visualizers
        self.traj_viz_depth = TrajViz(
            intrinsics=self.K_depth,
            cam_resolution=self.depth_img_shape,
        )
        self.traj_viz_rgb = TrajViz(
            intrinsics=self.K_bgr,
            cam_resolution=self.bgr_img_shape,
        )
        return

    def load_transforms(self):
        self.odom_rgb = np.loadtxt(os.path.join(self.dir, "odom_bgr.txt"))
        self.odom_depth = np.loadtxt(os.path.join(self.dir, "odom_depth.txt"))
        self.odom_sem = np.loadtxt(os.path.join(self.dir, "odom_sem.txt"))
        self.odom_goal = np.loadtxt(os.path.join(self.dir, "odom_goal.txt"))
        self.path = np.load(os.path.join(self.dir, "path.npy"))  # 3D

        self.depth_img_list = np.array(sorted(os.listdir(os.path.join(self.dir, "depth"))))
        self.bgr_img_list = np.array(sorted(os.listdir(os.path.join(self.dir, "bgr"))))
        self.sem_img_list = np.array(sorted(os.listdir(os.path.join(self.dir, "sem"))))

        assert len(self.depth_img_list) == len(self.odom_depth)
        assert len(self.bgr_img_list) == len(self.odom_rgb)
        assert len(self.sem_img_list) == len(self.odom_sem)

        return

    def plot(self):
        depth_time = self.odom_depth[:, 7] + self.odom_depth[:, 8] / 1e9
        goal_time = self.odom_goal[:, 3] + self.odom_goal[:, 4] / 1e9
        rgb_time = self.odom_rgb[:, 7] + self.odom_rgb[:, 8] / 1e9
        sem_time = self.odom_sem[:, 7] + self.odom_sem[:, 8] / 1e9
        path_time = self.path[:, 0, 3] + self.path[:, 0, 4] / 1e9

        os.makedirs(os.path.join(self.dir, "video_depth_projected"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "video_depth_iplanner_projected"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "video_rgb_projected"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "video_sem_projected"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "video_sem_after_projected"), exist_ok=True)

        # filter depth timestamps
        rgb_idx, depth_idx = RealWorldDataHandler.synchronize_data(
            self.odom_depth[:, 7:], self.odom_rgb[:, 7:], threshold=1.0
        )
        self.depth_img_list = self.depth_img_list[depth_idx]
        self.bgr_img_list = self.bgr_img_list[rgb_idx]
        self.odom_depth = self.odom_depth[depth_idx]
        self.odom_rgb = self.odom_rgb[rgb_idx]
        depth_time = depth_time[depth_idx]
        rgb_time = rgb_time[rgb_idx]

        # get iplanner model
        model_dir = "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models"
        self.net = torch.jit.load(os.path.join(model_dir, "plannernet_scripted.pt"))
        self.traj_generate = TrajOpt()

        # load data timestamps and synchroniize them
        for idx, path in enumerate(self.path):
            # goal_idx  = np.where(goal_time < path_time[idx])[0]
            # if len(goal_idx) == 0:
            #     goal_idx = 0
            # else:
            #     goal_idx = goal_idx[-1]
            goal_idx = 1

            # get depth images corresponding to current path
            if idx == len(self.path) - 1:
                depth_idx = np.where((depth_time >= path_time[idx]) & (depth_time <= path_time[idx] + 2))[0]
            else:
                depth_idx = np.where(
                    (depth_time >= path_time[idx])
                    & (depth_time <= path_time[idx + 1])
                    & (depth_time <= path_time[idx] + 2)
                )[
                    0
                ]  #

            if len(depth_idx) > 0:
                if depth_idx[0] < 810:
                    continue

                curr_depth_quat = tf.Rotation.from_matrix(
                    tf.Rotation.from_quat(self.odom_depth[depth_idx, 3:7]).as_matrix() @ ROS_TO_ROBOTICS_MAT
                ).as_quat()

                path_odom = path[None, :, :3].repeat(len(depth_idx), axis=0)
                goal_odom = self.odom_goal[None, goal_idx, :3].repeat(len(depth_idx), axis=0)

                depth_img = torch.zeros((len(depth_idx), self.depth_img_shape[0], self.depth_img_shape[1]))
                for local_idx, curr_depth_idx in enumerate(depth_idx):
                    depth_img[local_idx] = torch.from_numpy(
                        cv2.imread(
                            os.path.join(self.dir, "depth", self.depth_img_list[curr_depth_idx]), cv2.IMREAD_UNCHANGED
                        )
                        / 1000.0
                    )  # rotate by 180 deg

                projected_img_depth = self.traj_viz_depth.VizImages(
                    preds=torch.tensor(path_odom[:, [0, 9, 19, 29, 39, 49]], dtype=torch.float32),
                    waypoints=torch.tensor(path_odom, dtype=torch.float32),
                    odom=torch.tensor(
                        np.hstack((self.odom_depth[depth_idx, :3], curr_depth_quat)), dtype=torch.float32
                    ),
                    goal=torch.tensor(goal_odom, dtype=torch.float32),
                    fear=np.zeros((len(depth_idx), 1)),
                    images=depth_img,
                    is_shown=False,
                    transform=False,
                )

                for local_idx, curr_depth_idx in enumerate(depth_idx):
                    assert cv2.imwrite(
                        os.path.join(self.dir, "video_depth_projected", self.depth_img_list[curr_depth_idx]),
                        projected_img_depth[local_idx],
                    )

                # # apply iplanner
                # goal_depth_frame = (self.odom_goal[goal_idx, :3] - self.odom_depth[depth_idx[0], :3]) @ tf.Rotation.from_quat(curr_depth_quat[0]).as_matrix()
                # goal_depth_frame = torch.tensor(goal_depth_frame, dtype=torch.float32).unsqueeze(0).cuda()
                # depth_img_tensor = torch.tensor(cv2.resize(depth_img[0].numpy(), (640, 360), interpolation=cv2.INTER_CUBIC), dtype=torch.float32).unsqueeze(0)[None, :, :, :]
                # depth_img_tensor = depth_img_tensor.expand(-1, 3, -1, -1).cuda()

                # with torch.no_grad():
                #     keypoints, fear = self.net(depth_img_tensor, goal_depth_frame)
                # trajectory = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, 0.1)
                # trajectory_ws = TrajCost.TransformPoints(torch.tensor(np.hstack((self.odom_depth[depth_idx[0], :3], curr_depth_quat[0])), dtype=torch.float32).unsqueeze(0).cuda(), trajectory).tensor()
                # keypoints_ws = TrajCost.TransformPoints(torch.tensor(np.hstack((self.odom_depth[depth_idx[0], :3], curr_depth_quat[0])), dtype=torch.float32).unsqueeze(0).cuda(), keypoints).tensor()
                # fear = fear.cpu().numpy().repeat(len(depth_idx), axis=0)

                # projected_img_depth = self.traj_viz_depth.VizImages(
                #     preds=keypoints_ws.repeat(len(depth_idx), 1, 1),
                #     waypoints=trajectory_ws.repeat(len(depth_idx), 1, 1),
                #     odom=torch.tensor(np.hstack((self.odom_depth[depth_idx, :3], curr_depth_quat)), dtype=torch.float32),
                #     goal=torch.tensor(goal_odom, dtype=torch.float32),
                #     fear=fear,
                #     images=depth_img,
                #     is_shown=False,
                #     color_swap=True,
                #     transform=False,
                # )

                # for local_idx, curr_depth_idx in enumerate(depth_idx):
                #     assert cv2.imwrite(os.path.join(self.dir, "video_depth_iplanner_projected", self.depth_img_list[curr_depth_idx]), projected_img_depth[local_idx])

            # semantics
            # if idx == len(self.path) - 1:
            #     sem_idx = np.where((sem_time >= path_time[idx]) & (sem_time <= path_time[idx]+2))[0]
            # else:
            #     sem_idx = np.where((sem_time >= path_time[idx]) & (sem_time <= path_time[idx+1]) & (sem_time <= path_time[idx]+1))[0]

            # if len(sem_idx) > 0:
            #     curr_sem_quat = tf.Rotation.from_matrix(tf.Rotation.from_quat(self.odom_sem[sem_idx, 3:7]).as_matrix() @ ROS_TO_ROBOTICS_MAT).as_quat()

            #     path_odom = path[None, :, :3].repeat(len(sem_idx), axis=0)
            #     goal_odom = self.odom_goal[None, goal_idx, :3].repeat(len(sem_idx), axis=0)

            #     sem_img = torch.zeros((len(sem_idx), self.bgr_img_shape[2], self.bgr_img_shape[0], self.bgr_img_shape[1]))
            #     for local_idx, curr_sem_idx in enumerate(sem_idx):
            #         sem_img_curr         = cv2.imread(os.path.join(self.dir, "sem", self.sem_img_list[curr_sem_idx]))
            #         sem_img_curr         = cv2.resize(sem_img_curr, (self.bgr_img_shape[1], self.bgr_img_shape[0]), interpolation=cv2.INTER_CUBIC)
            #         sem_img[local_idx]   = torch.from_numpy(cv2.cvtColor(sem_img_curr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255.0

            #     projected_img_sem = self.traj_viz_rgb.VizImages(
            #         preds=torch.tensor(path_odom[:, [0, 9, 19, 29, 39, 49]], dtype=torch.float32),
            #         waypoints=torch.tensor(path_odom, dtype=torch.float32),
            #         odom=torch.tensor(np.hstack((self.odom_sem[sem_idx, :3], curr_sem_quat)), dtype=torch.float32),
            #         goal=torch.tensor(goal_odom, dtype=torch.float32),
            #         fear=np.zeros((len(sem_idx), 1)),
            #         images=sem_img,
            #         is_shown=False,
            #         transform=False,
            #     )

            #     for local_idx, curr_sem_idx in enumerate(sem_idx):
            #         assert cv2.imwrite(os.path.join(self.dir, "video_sem_projected", self.sem_img_list[curr_sem_idx]), projected_img_sem[local_idx])

            # rgb and semantics (after)
            if idx == len(self.path) - 1:
                rgb_idx = np.where((rgb_time >= path_time[idx]) & (rgb_time <= path_time[idx] + 2))[0]
            else:
                rgb_idx = np.where(
                    (rgb_time >= path_time[idx]) & (rgb_time <= path_time[idx + 1]) & (rgb_time <= path_time[idx] + 2)
                )[0]

            if len(rgb_idx) > 0:
                curr_rgb_quat = tf.Rotation.from_matrix(
                    tf.Rotation.from_quat(self.odom_rgb[rgb_idx, 3:7]).as_matrix() @ ROS_TO_ROBOTICS_MAT
                ).as_quat()

                path_odom = path[None, :, :3].repeat(len(rgb_idx), axis=0)
                goal_odom = self.odom_goal[None, goal_idx, :3].repeat(len(rgb_idx), axis=0)

                rgb_img = torch.zeros(
                    (len(rgb_idx), self.bgr_img_shape[2], self.bgr_img_shape[0], self.bgr_img_shape[1])
                )
                sem_img = torch.zeros(
                    (len(rgb_idx), self.bgr_img_shape[2], self.bgr_img_shape[0], self.bgr_img_shape[1])
                )
                for local_idx, curr_rgb_idx in enumerate(rgb_idx):
                    rgb_img_curr = cv2.imread(os.path.join(self.dir, "bgr", self.bgr_img_list[curr_rgb_idx]))
                    rgb_img[local_idx] = (
                        torch.from_numpy(cv2.cvtColor(rgb_img_curr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255.0
                    )
                    sem_img_curr = cv2.imread(os.path.join(self.dir, "semantics", self.bgr_img_list[curr_rgb_idx]))
                    rgb_sem_img = cv2.addWeighted(rgb_img_curr, 0.6, sem_img_curr, 0.4, 0)
                    sem_img[local_idx] = (
                        torch.from_numpy(cv2.cvtColor(rgb_sem_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255.0
                    )

                projected_img_rgb = self.traj_viz_rgb.VizImages(
                    preds=torch.tensor(path_odom[:, [0, 9, 19, 29, 39, 49]], dtype=torch.float32),
                    waypoints=torch.tensor(path_odom, dtype=torch.float32),
                    odom=torch.tensor(np.hstack((self.odom_rgb[rgb_idx, :3], curr_rgb_quat)), dtype=torch.float32),
                    goal=torch.tensor(goal_odom, dtype=torch.float32),
                    fear=np.zeros((len(rgb_idx), 1)),
                    images=rgb_img,
                    is_shown=False,
                    transform=False,
                )

                projected_sem_rgb = self.traj_viz_rgb.VizImages(
                    preds=torch.tensor(path_odom[:, [0, 9, 19, 29, 39, 49]], dtype=torch.float32),
                    waypoints=torch.tensor(path_odom, dtype=torch.float32),
                    odom=torch.tensor(np.hstack((self.odom_rgb[rgb_idx, :3], curr_rgb_quat)), dtype=torch.float32),
                    goal=torch.tensor(goal_odom, dtype=torch.float32),
                    fear=np.zeros((len(rgb_idx), 1)),
                    images=sem_img,
                    is_shown=False,
                    transform=False,
                )

                for local_idx, curr_rgb_idx in enumerate(rgb_idx):
                    assert cv2.imwrite(
                        os.path.join(self.dir, "video_rgb_projected", self.bgr_img_list[curr_rgb_idx]),
                        projected_img_rgb[local_idx],
                    )
                    assert cv2.imwrite(
                        os.path.join(self.dir, "video_sem_after_projected", self.bgr_img_list[curr_rgb_idx]),
                        projected_sem_rgb[local_idx],
                    )


if __name__ == "__main__":
    dir = "/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_stairs_both_door"
    # dir = "/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_crosswalk_sidewalk_success"
    # dir = "/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_crosswalk_sidewalk_wet_success"
    eval_plotter = EvalVideo(dir)
    eval_plotter.plot()
# EoF
