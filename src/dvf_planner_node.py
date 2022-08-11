#!/usr/bin/env python3

from ast import arg
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

# rospack = rospkg.RosPack()
# pack_path = rospack.get_path('dvf_planner_node')
# DEBUG
pack_path = os.getcwd()
planner_path = os.path.join(pack_path,'dvf_planner')
sys.path.append(pack_path)
sys.path.append(planner_path)

from rosutil import ROSArgparse
from dvf_planner import trajectory

class InterestNode:
    def __init__(self, args, node_name, transform):
        super(InterestNode, self).__init__()
        self.config(args, node_name)
        self.transform, self.bridge = transform, CvBridge()

        net, _ = torch.load(self.model_save)
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.image_time = rospy.get_rostime()
        self.traj_generate = trajectory.TrajOpt()
        self.is_goal_init = False
        
        rospy.Subscriber(self.image_topic, Image, self.imageCallback)
        rospy.Subscriber(self.goal_topic, PointStamped, self.goalCallback)
        # rospy.Subscriber(self.joy_topic, Joy, self.joyCallback)

        self.frame_pub = rospy.Publisher('path_image', Image, queue_size=10)
        self.path_pub  = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.tf_listener = tf.TransformListener()
        rospy.loginfo("DVF Planner Ready.")

    def joyCallback(self, msg):
        # TODO: generate goal from joystick command
        return
        

    def config(self, args, node_name):
        self.model_save = args.model_save
        # get params from ROS
        # prefix = "/" + node_name + "/"
        # self.image_topic = rospy.get_param(prefix + "depth_topic")
        # self.goal_topic  = rospy.get_param(prefix + "goal_topic")
        # self.path_topic  = rospy.get_param(prefix + "path_topic")
        # self.frame_id    = rospy.get_param(prefix + "robot_id")
        # DEBUG 
        self.image_topic = '/rgbd_camera/depth/image'
        self.goal_topic  = '/way_point'
        self.path_topic  = '/path'
        self.frame_id    = 'vehicle'
        return 

    def spin(self):
        rospy.spin()

    def pubPath(self, waypoint):
        path = Path()
        waypoint = waypoint.squeeze(0)
        for p in waypoint:
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
        return

    def imageCallback(self, msg):
        # rospy.loginfo("Received image %s: %d"%(msg.header.frame_id, msg.header.seq))
        self.image_time = msg.header.stamp
        try:
            frame = self.bridge.imgmsg_to_cv2(msg)
            frame = PIL.Image.fromarray(frame)
            frame = self.transform(frame)[None, ...]
        except CvBridgeError:
            rospy.logerr(CvBridgeError)
        else:
            if self.is_goal_init:
                p_in_vehicle = self.goal_pose
                p_in_vehicle.header.stamp = rospy.Time(0)
                try:
                    p_in_vehicle = self.tf_listener.transformPoint(self.frame_id, p_in_vehicle)
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr("Fail to transfer the goal into vehicle frame.")
                    return
                p_in_vehicle = torch.tensor([p_in_vehicle.point.x, p_in_vehicle.point.y, p_in_vehicle.point.z], dtype=torch.float32)
            else:
                return
            frame = frame.cuda() if torch.cuda.is_available() else frame
            goal  = p_in_vehicle.cuda() if torch.cuda.is_available() else p_in_vehicle
            with torch.no_grad():
                preds = self.net(frame, goal[None, ...])
                waypoint = self.traj_generate.TrajGeneratorFromPFreeRot(preds)
            self.pubPath(waypoint)

if __name__ == '__main__':

    node_name = "dvf_planner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument('model-save', type=str, default=planner_path+'/models/plannernet.pt', help="read model")
    parser.add_argument('crop-size', default=[360,640], help='image crop size')
    args = parser.parse_args()

    depth_transform = transforms.Compose([
        transforms.Resize(tuple(args.crop_size)),
        transforms.ToTensor()])

    node = InterestNode(args, node_name, depth_transform)

    node.spin()
