#!/usr/bin/env python3
import PIL
import torch
import torchvision.transforms as transforms

from viplanner import traj_opt

class VIPlannerAlgo:
    def __init__(self, args):
        super(VIPlannerAlgo, self).__init__()
        self.config(args)

        self.depth_transform = transforms.Compose([
            transforms.Resize(tuple(self.crop_size)),
            transforms.ToTensor()])

        net, _ = torch.load(self.model_save, map_location=torch.device("cpu"))
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.traj_generate = traj_opt.TrajOpt()
        return None

    def config(self, args):
        self.model_save = args.model_save
        self.crop_size  = args.crop_size
        return None


    def plan(self, image, goal_robot_frame):
        img = PIL.Image.fromarray(image)
        img = self.depth_transform(img)
        if torch.cuda.is_available():
            img = img.cuda()
            goal_robot_frame = goal_robot_frame.cuda()
        with torch.no_grad():
            keypoints, fear = self.net(img, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints , step=0.1)
        
        return keypoints, traj, fear, img
