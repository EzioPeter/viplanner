#!/usr/bin/env python3

import os
import PIL
import sys
import torch
import rospy
import rospkg
import tf
import numpy as np
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

from dvf_planner.rosutil import ROSArgparse
from dvf_planner import trajectory
from dvf_planner import visualizer

class InterestNode:
    def __init__(self, args, transform):
        super(InterestNode, self).__init__()
        self.config(args)
        self.transform, self.bridge = transform, CvBridge()

        net, _ = torch.load(self.model_save)
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.traj_generate = trajectory.TrajOpt()
        self.visualizer = visualizer.TrajViz(os.path.join(*[planner_path, 'data', "robot"]), "robot")

        self.image_time = rospy.get_rostime()
        self.is_goal_init = False
        self.ready_for_visual = False
        
        self.traj_generate = trajectory.TrajOpt()
        
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
        

    def config(self, args):
        self.is_visual   = args.is_visual
        self.render_freq = args.render_freq
        self.model_save  = args.model_save
        self.image_topic = args.depth_topic
        self.goal_topic  = args.goal_topic
        self.path_topic  = args.path_topic
        self.frame_id    = args.robot_id
        return 

    def spin(self):
        r = rospy.Rate(self.render_freq)
        while not rospy.is_shutdown():
            if self.is_visual and self.ready_for_visual:
                self.pubRenderImage(self.preds, self.waypoints, self.odom, self.goal, self.frame)
            r.sleep()
        rospy.spin()

    def pubPath(self, waypoints):
        path = Path()
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

    def pubRenderImage(self, preds, waypoints, odom, goal, image):
        cv_image = self.visualizer.VizImages(preds, waypoints, odom, goal, image, is_shown=False)[0]
        cv_ros = self.bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        self.frame_pub.publish(cv_ros)
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
            frame = np.array(self.bridge.imgmsg_to_cv2(msg))
            frame[~np.isfinite(frame)] = 0.0
            frame = PIL.Image.fromarray(frame)
            # DEBUG - Visual Image
            # img = PIL.Image.fromarray((frame * 255 / np.max(frame[frame>0])).astype('uint8'))
            # img.show()
            frame = self.transform(frame)[None, ...]
        except CvBridgeError:
            rospy.logerr(CvBridgeError)
        else:
            if self.is_goal_init:
                p_in_vehicle = self.goal_pose
                p_in_vehicle.header.stamp = rospy.Time(0)
                try:
                    p_in_vehicle = self.tf_listener.transformPoint(self.frame_id, p_in_vehicle)
                    (odom, ori) = self.tf_listener.lookupTransform('map', self.frame_id, rospy.Time(0))
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
                self.frame = frame.cuda()
                self.goal  = p_in_vehicle.cuda()
            else:
                self.odom  = odom, self.frame = frame, self.goal  = p_in_vehicle
            
            with torch.no_grad():
                self.preds = self.net(self.frame, self.goal)
                self.waypoints = self.traj_generate.TrajGeneratorFromPFreeRot(self.preds)
            self.pubPath(self.waypoints)
            self.ready_for_visual = True
        return

if __name__ == '__main__':

    node_name = "dvf_planner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument('is_visual',    type=bool,  default=True,                          help="frequence for path image rendering")
    parser.add_argument('render_freq',  type=int,   default=5,                          help="frequence for path image rendering")
    parser.add_argument('model_save',   type=str,   default='/models/plannernet.pt',    help="read model")
    parser.add_argument('crop_size',    type=tuple, default=[360,640],                  help='image crop size')
    parser.add_argument('depth_topic',  type=str,   default='/rgbd_camera/depth/image', help='depth image ros topic')
    parser.add_argument('goal_topic',   type=str,   default='/way_point',               help='goal waypoint ros topic')
    parser.add_argument('path_topic',   type=str,   default='/path',                    help='DVF Path topic')
    parser.add_argument('robot_id',     type=str,   default='vehicle',                  help='DVF Path topic')
    
    args = parser.parse_args()
    args.model_save = planner_path + args.model_save

    depth_transform = transforms.Compose([
        transforms.Resize(tuple(args.crop_size)),
        transforms.ToTensor()])

    node = InterestNode(args, depth_transform)

    node.spin()
