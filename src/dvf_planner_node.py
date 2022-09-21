#!/usr/bin/env python3
import os
import PIL
import sys
import torch
import rospy
import rospkg
import tf
import time
from std_msgs.msg import Float32
import numpy as np
from sensor_msgs.msg import Image
import torchvision.transforms as transforms
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
import ros_numpy

rospack = rospkg.RosPack()
pack_path = rospack.get_path('dvf_planner_node')
planner_path = os.path.join(pack_path,'dvf_planner')
sys.path.append(pack_path)
sys.path.append(planner_path)

from dvf_planner.rosutil import ROSArgparse
from dvf_planner import traj_opt

class InterestNode:
    def __init__(self, args, transform):
        super(InterestNode, self).__init__()
        self.config(args)
        self.transform = transform

        net, _ = torch.load(self.model_save, map_location=torch.device("cpu"))
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.traj_generate = traj_opt.TrajOpt()

        self.image_time = rospy.get_rostime()
        self.is_goal_init = False
        self.ready_for_planning = False
        self.is_goal_processed = False
        self.timer_data = Float32()
        
        rospy.Subscriber(self.image_topic, Image, self.imageCallback)
        rospy.Subscriber(self.goal_topic, PointStamped, self.goalCallback)

        timer_topic = '/dvf_timer'
        self.timer_pub = rospy.Publisher(timer_topic, Float32, queue_size=10)
        self.path_pub  = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.tf_listener = tf.TransformListener()
        rospy.loginfo("DVF Planner Ready.")
        

    def config(self, args):
        self.main_freq   = args.main_freq
        self.model_save  = args.model_save
        self.image_topic = args.depth_topic
        self.goal_topic  = args.goal_topic
        self.path_topic  = args.path_topic
        self.frame_id    = args.robot_id
        self.world_id    = args.world_id
        self.uint_type   = args.uint_type
        self.image_flip  = args.image_flip
        self.conv_dist   = args.conv_dist  
        return 

    def spin(self):
        r = rospy.Rate(self.main_freq)
        while not rospy.is_shutdown():
            if self.ready_for_planning:
                # main planning starts
                start = time.time()
                with torch.no_grad():
                    self.preds = self.net(self.img, self.goal)
                    self.waypoints = self.traj_generate.TrajGeneratorFromPFreeRot(self.preds)
                end = time.time()
                self.timer_data.data = (end - start) * 1000
                self.timer_pub.publish(self.timer_data)
                # check goal less than converage range
                goal_np = self.goal[0, :].cpu().detach().numpy()
                if (np.sqrt(goal_np[0]**2 + goal_np[1]**2) < self.conv_dist) and self.is_goal_processed:
                    self.ready_for_planning = False
                    self.is_goal_init = False
                    rospy.loginfo("Goal Arrived")
                self.pubPath(self.waypoints, self.is_goal_init)
            r.sleep()
        rospy.spin()

    def pubPath(self, waypoints, is_goal_init=True):
        path = Path()
        if is_goal_init:
            waypoints = waypoints.squeeze(0)
            for p in waypoints:
                pose = PoseStamped()
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                path.poses.append(pose)
        # add header
        path.header.frame_id = self.frame_id
        path.header.stamp = self.image_time
        self.path_pub.publish(path)
        return

    def goalCallback(self, msg):
        rospy.loginfo("Recevied a new goal")
        self.goal_pose = msg
        self.is_goal_init = True
        self.is_goal_processed = False
        return

    def imageCallback(self, msg):
        # rospy.loginfo("Received image %s: %d"%(msg.header.frame_id, msg.header.seq))
        self.image_time = msg.header.stamp
        frame = ros_numpy.numpify(msg)
        frame[~np.isfinite(frame)] = 0
        if self.uint_type:
            frame = frame / 1000.0
        # DEBUG - Visual Image
        # img = PIL.Image.fromarray((frame * 255 / np.max(frame[frame>0])).astype('uint8'))
        # img.show()
        img = PIL.Image.fromarray(frame)
        if self.image_flip:
            img = img.transpose(PIL.Image.ROTATE_180)
        img = self.transform(img)[None, ...]
        if self.is_goal_init:
            p_in_vehicle = self.goal_pose
            p_in_vehicle.header.stamp = rospy.Time(0)
            try:
                p_in_vehicle = self.tf_listener.transformPoint(self.frame_id, p_in_vehicle)
                (odom, ori) = self.tf_listener.lookupTransform(self.world_id, self.frame_id, rospy.Time(0))
                odom.extend(ori)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logerr("Fail to transfer the goal into vehicle frame.")
                return
            p_in_vehicle = torch.tensor([p_in_vehicle.point.x, p_in_vehicle.point.y, p_in_vehicle.point.z], dtype=torch.float32)[None, ...]
        else:
            return
        odom = torch.tensor(odom, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            self.odom  = odom.cuda()
            self.img   = img.cuda()
            self.goal  = p_in_vehicle.cuda()
        else:
            self.odom  = odom, self.im = img, self.goal = p_in_vehicle
        self.ready_for_planning = True
        self.is_goal_processed = True
        return

if __name__ == '__main__':

    node_name = "dvf_planner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument('main_freq',     type=int,   default=5,                          help="frequency of path planner")
    parser.add_argument('model_save',    type=str,   default='/models/plannernet.pt',    help="read model")
    parser.add_argument('crop_size',     type=tuple, default=[360,640],                  help='image crop size')
    parser.add_argument('uint_type',     type=bool,  default=False,                      help="image in uint type or not")
    parser.add_argument('depth_topic',   type=str,   default='/rgbd_camera/depth/image', help='depth image ros topic')
    parser.add_argument('goal_topic',    type=str,   default='/way_point',               help='goal waypoint ros topic')
    parser.add_argument('path_topic',    type=str,   default='/path',                    help='DVF Path topic')
    parser.add_argument('robot_id',      type=str,   default='base',                     help='robot TF frame id')
    parser.add_argument('world_id',      type=str,   default='odom',                     help='world TF frame id')
    parser.add_argument('image_flip',    type=bool,  default=True,                       help='is the image fliped')
    parser.add_argument('conv_dist',     type=float, default=0.5,                        help='converge range to the goal')

    args = parser.parse_args()
    args.model_save = planner_path + args.model_save

    depth_transform = transforms.Compose([
        transforms.Resize(tuple(args.crop_size)),
        transforms.ToTensor()])

    node = InterestNode(args, depth_transform)

    node.spin()
