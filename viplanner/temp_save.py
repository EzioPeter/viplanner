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
    def __init__(self, in_channels, goal_channels, k=5, fear_channels=64):
        super().__init__()
        self.k = k
        self.relu    = nn.LeakyReLU(inplace=True)
        self.fg      = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0);
        self.conv3 = nn.Conv2d(256+fear_channels, 16, kernel_size=3, stride=1, padding=1)

        self.fc1   = nn.Linear(256 * 128, 1024) 
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512,  k*3)
        
        self.ff = nn.Linear(k*3, fear_channels)
        self.lossfc = nn.Linear(2048, 1)

    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)
        # compute x
        x = self.relu(self.conv1(x))
        f = self.relu(self.conv2(x))
        
        x = torch.flatten(f, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        c = self.ff(x)
        c = c[:, :, None, None].expand(-1, -1, f.shape[2], f.shape[3])
        c = torch.cat((f, c), dim=1)
        c = self.relu(self.conv3(c))
        c = torch.flatten(c, 1)

        x = x.reshape(-1, self.k, 3)
        c = self.sigmoid(self.lossfc(c))
        return x, c


if __name__ == "__main__":
    from dataset import PlannerData
    from torchutil import show_batch
    import torchvision.transforms as transforms

    depth_transform = transforms.Compose([
            # transforms.RandomRotation(20),
            transforms.Resize(tuple([640,480])),
            transforms.ToTensor()])

    
    batch_size = 2
    root_path = os.getcwd() + "/data/"
    data = PlannerData(root=root_path, train=True, transform=depth_transform)
    loader = Data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

    knode = 10
    net = AutoEncoder("resnet", knode)
    if torch.cuda.is_available():
        net = net.cuda()

    with torch.no_grad():
        enumerater = tqdm.tqdm(enumerate(loader))
        for batch_idx, inputs in enumerater:
            if torch.cuda.is_available():
                image = inputs[0].cuda()
                odom  = inputs[1].tensor().cuda()
                goal  = inputs[2].tensor().cuda()
            outputs, flags = net(image, goal)
            show_batch(torch.cat([inputs, outputs], dim=0), name='test', waitkey=1000)
