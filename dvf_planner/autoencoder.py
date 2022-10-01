#!/usr/bin/env python3
import os
import tqdm
import torch
from PlannerNet import PlannerNet
import torch.nn as nn
import torch.utils.data as Data


class AutoEncoder(nn.Module):
    def __init__(self, encoder_channel=64, k=5):
        super().__init__()
        self.encoder = PlannerNet(layers=[2, 2, 2, 2])
        self.decoder = Decoder(512, encoder_channel, k)

    def forward(self, x, goal):
        x = x.expand(-1, 3, -1, -1)
        x = self.encoder(x)
        x, c = self.decoder(x, goal)
        return x, c


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu    = nn.LeakyReLU(inplace=True)
        self.fg      = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0);

        self.fc1   = nn.Linear(256 * 128, 1024) 
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512,  k*3)
        
        self.lossfc = nn.Linear(1024, 1)

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

        c = self.sigmoid(self.lossfc(f))
        return x, c
