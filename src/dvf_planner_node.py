#!/usr/bin/env python3

import os
import PIL
import sys
import torch
import rospy
import rospkg
import tf
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torchvision.transforms as transforms
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped

rospack = rospkg.RosPack()
pack_path = rospack.get_path('dvf_planner_node')
planner_path = os.path.join(pack_path,'dvf_planner')
sys.path.append(pack_path)
sys.path.append(planner_path)

from rosutil import ROSArgparse
from dvf_planner import trajectory

class InterestNode:
    def __init__(self, node_name, transform):
        super(InterestNode, self).__init__()
        self.config(node_name)
        self.transform, self.bridge = transform, CvBridge()

        net = torch.load(self.model_save)
        net.set_train(False)
        self.image_time = rospy.get_rostime()
        self.net = net.cuda() if torch.cuda.is_available() else net
        self.traj_generate = trajectory.TrajOpt()
        self.is_goal_init = False
        self.goal = PointStamped()
        
        rospy.Subscriber(self.image_topic, Image, self.imageCallback)
        rospy.Subscriber(self.goal_topic, PointStamped, self.goalCallback)
        rospy.Subscriber(self.joy_topic, Joy, self.joyCallback)

        self.frame_pub = rospy.Publisher('path_image', Image, queue_size=10)
        self.path_pub  = rospy.Publisher('/path', Path, queue_size=10)
        self.tf_listener = tf.TransformListener()

    def joyCallback(self, msg):
        # generate goal from joystick command
        # joyTime = ros::Time::now().toSec();

        # joySpeedRaw = sqrt(joy->axes[3] * joy->axes[3] + joy->axes[4] * joy->axes[4]);
        # joySpeed = joySpeedRaw;
        # if (joySpeed > 1.0) joySpeed = 1.0;
        # if (joy->axes[4] == 0) joySpeed = 0;

        # if (joySpeed > 0) {
        #     joyDir = atan2(joy->axes[3], joy->axes[4]) * 180 / PI;
        #     if (joy->axes[4] < 0) joyDir *= -1;
        # }

        # if (joy->axes[4] < 0 && !twoWayDrive) joySpeed = 0;

        # if (joy->axes[2] > -0.1) {
        #     autonomyMode = false;
        # } else {
        #     autonomyMode = true;
        # }

        # if (joy->axes[5] > -0.1) {
        #     checkObstacle = true;
        # } else {
        #     checkObstacle = false;
        # }
        return
        

    def config(self, node_name):
        # get params from ROS
        prefix = "/" + node_name + "/"
        self.image_topic = rospy.get_param(prefix + "depth_topic")
        self.goal_topic  = rospy.get_param(prefix + "goal_topic")
        self.frame_id    = rospy.get_param(prefix + "robot_id")
        
        # DEBUG 
        self.image_topic = '/rgbd_camera/depth/image'
        self.goal_topic  = '/mp_waypoint'
        self.frame_id    = 'vehicle'
        return 

    def spin(self):
        rospy.spin()

    def pubPath(self, waypoint):
        path = Path()
        for p in waypoint:
            pose = PoseStamped()
            pose.pose.position.x = p.x
            pose.pose.position.y = p.y
            pose.pose.position.z = p.z
            path.poses.append(pose)
        # add header
        path.header.frame_id = self.frame_id
        path.header.stamp = self.image_time
        self.path_pub(path)
        return

    def goalCallback(self, msg):
        rospy.loginfo("Recevied a new goal")
        p_in_vehicle = msg
        if msg.header.frame_id != self.frame_id:
            try:
                self.listener.waitForTransform(msg.header.frame_id, self.frame_id, rospy.Time(), rospy.Duration(1.0))
                p_in_vehicle = self.tf_listener.transformPoint(self.frame_id, msg)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("Fail to transfer the goal into vehicle frame.")
        self.is_goal_init = True
        self.goal_pose = torch.tensor([p_in_vehicle.point.x, p_in_vehicle.point.y, p_in_vehicle.point.z])
        return

    def imageCallback(self, msg):
        rospy.loginfo("Received image %s: %d"%(msg.header.frame_id, msg.header.seq))
        self.image_time = msg.header.stamp
        try:
            frame = self.bridge.imgmsg_to_cv2(msg)
            frame = PIL.Image.fromarray(frame)
            image = self.transform(frame)
            frame = self.normalize(image).unsqueeze(dim=0)
        except CvBridgeError:
            rospy.logerr(CvBridgeError)
        else:
            frame = frame.cuda() if torch.cuda.is_available() else frame
            preds = self.net(frame, self.goal)
            waypoint = self.traj_generate.TrajGeneratorFromPFreeRot(preds)
            self.pubPath(waypoint)

if __name__ == '__main__':

    node_name = "dvf_planner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument("model-save", type=str, default=pack_path+'/models/plannernet.pt', help="read model")
    parser.add_argument('--crop-size', nargs='+', type=int, default=[360,640], help='image crop size')
    args = parser.parse_args()

    depth_transform = transforms.Compose([
        transforms.Resize(tuple(args.crop_size)),
        transforms.ToTensor()])

    node = InterestNode(args, depth_transform)

    node.spin()
