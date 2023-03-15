#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import pickle
from typing import Optional
import argparse

# detectron2 and mask2former (used to load pre-trained models from Mask2Former)
try:
    os.environ["DETECTRON2_DISABLE_CV2"] = "1"
    from detectron2.modeling.backbone import build_resnet_backbone
    from detectron2.config import get_cfg, CfgNode
    from detectron2.projects.deeplab import add_deeplab_config
    from third_party.mask2former.mask2former import add_maskformer2_config
    pre_train_possible = True
except ImportError:
    pre_train_possible = False
    print("[Warning] Pre-trained ResNet50 models cannot be used since detectron2 and/or mask2former not found")

# visual-imperative-planner
from .PlannerNet import PlannerNet
from config import TrainCfg

def get_m2f_cfg(cfg_path: str) -> CfgNode:
    # load config from file
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg

  
class RGBEncoder(nn.Module):
    def __init__(self, cfg: CfgNode, weight_path: Optional[str] = None, freeze: bool = True) -> None:
        super().__init__()
        
        # load pre-trained resnet
        input_shape = argparse.Namespace()
        input_shape.channels = 3
        self.backbone = build_resnet_backbone(cfg, input_shape)
        
        # load weights
        if weight_path is not None:
            with open(weight_path, "rb") as file:
                model_file = pickle.load(file, encoding="latin1")
            
            model_file['model'] = {k.replace("backbone.", ""): torch.tensor(v) for k, v in model_file['model'].items()}
                    
            missing_keys, unexpected_keys = self.backbone.load_state_dict(model_file['model'], strict=False)
            if len(missing_keys) != 0:
                print(f"[WARNING] Missing keys: {missing_keys}")
                print(f"[WARNING] Unexpected keys: {unexpected_keys}")
            print(f"[INFO] Loaded pre-trained backbone from {weight_path}")
        
        # freeze network
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # layers to get correct output shape --> modifiable
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        
        return
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)['res5']    # size = (N, 2048, 12, 20) (height and width same as ResNet18)
        x = self.conv1(x)               # size = (N, 512,  12, 20) 
        return x
        
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder_channel=64, k=5):
        super().__init__()
        self.encoder = PlannerNet(layers=[2, 2, 2, 2])
        self.decoder = Decoder(512, encoder_channel, k)

    def forward(self, x: torch.Tensor, goal: torch.Tensor):
        x = x.expand(-1, 3, -1, -1)
        x = self.encoder(x)
        x, c = self.decoder(x, goal)
        return x, c


class DualAutoEncoder(nn.Module):
    def __init__(self, train_cfg: TrainCfg, m2f_cfg: Optional[CfgNode] = None, weight_path: Optional[str] = None):
        super().__init__()
        self.encoder_depth = PlannerNet(layers=[2, 2, 2, 2])
        if train_cfg.rgb and train_cfg.pre_train_sem and pre_train_possible:
            self.encoder_sem = RGBEncoder(m2f_cfg, weight_path, freeze=train_cfg.pre_train_freeze)
        else:
            self.encoder_sem = PlannerNet(layers=[2, 2, 2, 2])
        self.decoder = Decoder(1024, train_cfg.in_channel, train_cfg.knodes)
        return

    def forward(self, x_depth: torch.Tensor, x_sem: torch.Tensor, goal: torch.Tensor):
        # encode depth
        x_depth = x_depth.expand(-1, 3, -1, -1)
        x_depth = self.encoder_depth(x_depth)
        # encode sem
        x_sem = self.encoder_sem(x_sem)
        # concat
        x = torch.cat((x_depth, x_sem), dim=1)
        # decode
        x, c = self.decoder(x, goal)
        return x, c


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu    = nn.ReLU(inplace=True)
        self.fg      = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0);

        self.fc1   = nn.Linear(256 * 128, 1024) 
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512,  k*3)
        
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)
        # compute x
        x = self.relu(self.conv1(x))   # size = (N, 512, x.H/32, x.W/32)
        x = self.relu(self.conv2(x))   # size = (N, 512, x.H/60, x.W/60)
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        x = self.relu(self.fc2(f))
        x = self.fc3(x)
        x = x.reshape(-1, self.k, 3)

        c = self.relu(self.frc1(f))
        c = self.sigmoid(self.frc2(c))

        return x, c

# EoF
