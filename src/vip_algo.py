#!/usr/bin/env python3
import PIL
import torch
import torchvision.transforms as transforms

from dvf_planner import traj_opt

class VIPlannerAlgo:
    def __init__(self, args):
        super(VIPlannerAlgo, self).__init__()
        self.config(args)

        self.depth_transform = transforms.Compose([
            transforms.Resize(tuple(360, 640)),
            transforms.ToTensor()])

        net, _ = torch.load(self.model_save, map_location=torch.device("cpu"))
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.traj_generate = traj_opt.TrajOpt()
        return None

    def config(self, args):
        self.model_save = args.model_save
        return None


    def plan(self, image, goal_robot_frame):
        img = PIL.Image.fromarray(image)
        img = self.depth_transform(img)
        goal_robot_frame = torch.tensor(goal_robot_frame, dtype=torch.float32)[None, ...]
        if torch.cuda.is_available():
            img = img.cuda()
            goal_robot_frame = goal_robot_frame.cuda()
        with torch.no_grad:
            keypoints, fear = self.net(img, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints , step=0.1)
        
        return keypoints, traj, fear
