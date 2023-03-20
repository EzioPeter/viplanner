"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch
@author     Fan Yang
@email      fanyang1@ethz.ch


@brief      Visual Imperative Planner (VIPlanner) Inference Script
"""

# python
import os
import torch
import numpy as np
import math
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# viplanner src
from .viplanner.plannernet import DualAutoEncoder, get_m2f_cfg
from .viplanner.traj_cost_opt.traj_opt import TrajOpt
from .viplanner.config.learning_cfg import TrainCfg

torch.set_default_dtype(torch.float32)

   
class VIPlannerInference:
    def __init__(
        self,
        cfg,
    ) -> None:
        """ VIPlanner Inference Script

        Args:
            model_save (str): path to the model directory containing model.pt and model.yaml
            sensor_offset_x (float, optional): sensor offset in x direction. Defaults to 0.47.
            sensor_offset_y (float, optional): sensor offset in y direction. Defaults to 0.0.
        """
        # get configs
        self._sensor_offset_x = cfg.sensor_offset_x
        self._sensor_offset_y = cfg.sensor_offset_y
        model_path  = os.path.join(cfg.model_save, "model.pt")
        config_path = os.path.join(cfg.model_save, "model.yaml")
        
        # get train config
        self.train_cfg: TrainCfg = TrainCfg.from_yaml(config_path)
        
        # get model
        if self.train_cfg.rgb:
            m2f_cfg = get_m2f_cfg(cfg.m2f_config_path)
            self.pixel_mean = m2f_cfg.MODEL.PIXEL_MEAN
            self.pixel_std = m2f_cfg.MODEL.PIXEL_STD
        else:
            m2f_cfg = None
            self.pixel_mean = [0, 0, 0]
            self.pixel_std = [1, 1, 1]
        self.net = DualAutoEncoder(train_cfg=self.train_cfg, m2f_cfg=m2f_cfg)
        try:
            model_state_dict, _ = torch.load(model_path)
        except ValueError:
            model_state_dict = torch.load(model_path)
        self.net.load_state_dict(model_state_dict)

        # inference script = no grad for model
        self.net.eval()

        # move to GPU if available
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self._device = "cuda"
        else:
            self._device = "cpu"
            
        # transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(tuple(self.train_cfg.img_input_size))
            ]
        )
        
        # get trajectory generator
        self.traj_generate = TrajOpt()
        
        # recognize shift in trajectory
        self.is_traj_shift = False
        if math.hypot(self._sensor_offset_x, self._sensor_offset_y) > 1e-1:
            self.is_traj_shift = True
            print("Trajectory will be shifted by ({}, {})".format(self._sensor_offset_x, self._sensor_offset_y))
        return

    def img_converter(self, img: np.ndarray) -> torch.Tensor:
        # crop image and convert to tensor
        img = self.transforms(img)
        img = img.unsqueeze(0).to(self._device)
        return img
        
    def plan(
        self,
        depth_image: np.ndarray,
        sem_rgb_image: np.ndarray,
        goal_robot_frame: torch.Tensor,
    ) -> tuple:
        """Plan to path towards the goal given depth and semantic image

        Args:
            depth_image (np.ndarray): Depth image from the robot
            goal_robot_frame (torch.Tensor): Goal in robot frame
            sem_rgb_image (np.ndarray): Semantic/ RGB Image from the robot.

        Returns:
            tuple: _description_
        """
        # get keypoints and fear from planner
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(depth_image)
        # axs[1].imshow(sem_rgb_image)
        # fig.savefig("/root/git/network_input.png")
        # plt.close()
        # print(goal_robot_frame)
        
        with torch.no_grad():
            depth_image = self.img_converter(depth_image).float()
            if self.train_cfg.rgb:
                sem_rgb_image = (sem_rgb_image - self.pixel_mean) / self.pixel_std
            sem_rgb_image = self.img_converter(sem_rgb_image.astype(np.uint8)).float()
            keypoints, fear = self.net(depth_image, sem_rgb_image, goal_robot_frame.to(self._device))

        # add potential offset from sensor to robot
        if self.is_traj_shift:
            batch_size, _, dims = keypoints.shape
            keypoints = torch.cat((torch.zeros(batch_size, 1, dims, device=keypoints.device, requires_grad=False), keypoints), axis=1)
            keypoints[..., 0] += self._sensor_offset_x
            keypoints[..., 1] += self._sensor_offset_y
        
        # generate trajectory
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

# EoF
