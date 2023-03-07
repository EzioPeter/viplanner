"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch
@author     Fan Yang
@email      fanyang1@ethz.ch


@brief      Visual Imperative Planner (VIPlanner) Inference Script
"""

# python
import os
import PIL
import torch
import numpy as np
import math
from typing import Optional
import torchvision.transforms as transforms

# viplanner src
from .viplanner.plannernet.autoencoder import DualAutoEncoder
from .viplanner.traj_cost_opt.traj_opt import TrajOpt
from .viplanner.config.learning_cfg import TrainCfg

torch.set_default_dtype(torch.float32)

   
class VIPlannerInference:
    def __init__(
        self,
        model_save: str, 
        sensor_offset_x: float = 0.47,
        sensor_offset_y: float = 0.0,
    ) -> None:
        """ VIPlanner Inference Script

        Args:
            model_save (str): path to the model directory containing model.pt and model.yaml
            sensor_offset_x (float, optional): sensor offset in x direction. Defaults to 0.47.
            sensor_offset_y (float, optional): sensor offset in y direction. Defaults to 0.0.
        """
        # get configs
        self._sensor_offset_x = sensor_offset_x
        self._sensor_offset_y = sensor_offset_y
        model_path = os.path.join(model_save, "model.pt")
        config_path = os.path.join(model_save, "model.yaml")
        
        # get train config
        train_cfg: TrainCfg = TrainCfg.from_yaml(config_path)
        
        # get model
        self.net = DualAutoEncoder(encoder_channel=train_cfg.in_channel, k=train_cfg.knodes)
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
            transforms.Resize(tuple(train_cfg.img_input_size)),
            transforms.ToTensor()]
        )
        
        # get trajectory generator
        self.traj_generate = TrajOpt()
        
        # recognize shift in trajectory
        self.is_traj_shift = False
        if math.hypot(self._sensor_offset_x, self._sensor_offset_y) > 1e-1:
            self.is_traj_shift = True
        return

    def img_converter(self, img: np.ndarray) -> torch.Tensor:
        # crop image and convert to tensor
        img = PIL.Image.fromarray(img)
        img = self.transforms(img)
        img = img.unsqueeze(0).to(self._device)
        return img
        
    def plan(
        self,
        depth_image: np.ndarray,
        sem_image: np.ndarray,
        goal_robot_frame: torch.Tensor,
    ) -> tuple:
        """Plan to path towards the goal given depth and semantic image

        Args:
            depth_image (np.ndarray): Depth image from the robot
            goal_robot_frame (torch.Tensor): Goal in robot frame
            sem_image (Optional[np.ndarray], optional): Semantic Image from the robot. Defaults to None.

        Returns:
            tuple: _description_
        """
        # get keypoints and fear from planner
        with torch.no_grad():
            depth_image = self.img_converter(depth_image)
            sem_image = self.img_converter(sem_image)
            keypoints, fear = self.net(depth_image, sem_image, goal_robot_frame.to(self._device))

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
