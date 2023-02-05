#!/usr/bin/env python3
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch
@author     Fan Yang
@email      fanyang1@ethz.ch


@brief      Visual Imperative Planner (VIPlanner) ROS Node
"""

# python
import os
import PIL
import sys
import torch
import copy
import time
import numpy as np

# ROS
import rospy
import rospkg
import tf
from std_msgs.msg import Float32, Int16
from sensor_msgs.msg import Image, Joy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
import ros_numpy
import message_filters

# init ros node
rospack = rospkg.RosPack()
pack_path = rospack.get_path('viplanner_node')
planner_path = os.path.join(pack_path,'viplanner')
sys.path.append(pack_path)
sys.path.append(planner_path)

# visual imperative planner
from model_src.vip_inference import VIPlannerInference
from model_src.m2f_inference import M2FInference
from utils.rosutil import ROSArgparse


class VIPlannerNode:
    """VIPlanner ROS Node Class"""
    def __init__(self, args):
        super(VIPlannerNode, self).__init__()
        self.config(args)

        # init planner algo class
        self.vip_algo = VIPlannerInference(
            model_save=args.model_save,
            crop_size=args.crop_size,
            sensor_offset_x=args.sensor_offset_x,
            sensor_offset_y=args.sensor_offset_y,
        )
        
        self.tf_listener = tf.TransformListener()

        self.image_time = rospy.get_rostime()
        self.is_goal_init = False
        self.ready_for_planning = False

        # planner status
        self.planner_status = Int16()
        self.planner_status.data = 0
        self.is_goal_processed = False
        self.is_smartjoy = False

        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        # process time
        self.timer_data = Float32()
        
        # depth and rgb image message --> time syncronization by message_filters
        img_depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        img_sem_sub = message_filters.Subscriber(self.sem_topic, Image)
        ts = message_filters.TimeSynchronizer([img_depth_sub, img_sem_sub], 10)
        ts.registerCallback(self.imageCallback)
        
        # subscribe to further topics
        rospy.Subscriber(self.goal_topic, PointStamped, self.goalCallback)
        rospy.Subscriber("/joy", Joy, self.joyCallback, queue_size=10)

        timer_topic = '/vip_timer'
        status_topic = '/vip_planner_status'
        
        # planning status topics
        self.timer_pub = rospy.Publisher(timer_topic, Float32, queue_size=10)
        self.status_pub = rospy.Publisher(status_topic, Int16, queue_size=10)

        self.path_pub  = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.fear_path_pub = rospy.Publisher(self.path_topic + "_fear", Path, queue_size=10)

        rospy.loginfo("VIPlanner Ready.")
        

    def config(self, args):
        self.main_freq   = args.main_freq

        self.frame_id    = args.robot_id
        self.world_id    = args.world_id
        self.uint_type   = args.uint_type
        self.image_flip  = args.image_flip
        self.conv_dist   = args.conv_dist
        self.depth_max   = args.depth_max
        # ROS topics
        self.depth_topic = args.depth_topic
        self.rgb_topic   = args.rgb_topic
        self.goal_topic  = args.goal_topic
        self.path_topic  = args.path_topic        
        # fear reaction
        self.is_fear_act = args.is_fear_act
        self.buffer_size = args.buffer_size
        self.ang_thred   = args.angular_thred
        self.track_dist  = args.track_dist
        self.joyGoal_scale = args.joyGoal_scale
        return 

    def spin(self):
        r = rospy.Rate(self.main_freq)
        while not rospy.is_shutdown():
            if self.ready_for_planning and self.is_goal_init:
                # main planning starts
                cur_depth_image = self.depth_img.copy()
                cur_sem_image = self.sem_img.copy()
                start = time.time()
                # Network Planning
                self.preds, self.waypoints, self.fear, _ = self.vip_algo.plan(cur_depth_image, cur_sem_image, self.goal_rb)
                end = time.time()
                self.timer_data.data = (end - start) * 1000
                self.timer_pub.publish(self.timer_data)
                # check goal less than converage range
                if (np.sqrt(self.goal_rb[0][0]**2 + self.goal_rb[0][1]**2) < self.conv_dist) and self.is_goal_processed and (not self.is_smartjoy):
                    self.ready_for_planning = False
                    self.is_goal_init = False
                    # planner status -> Success
                    if self.planner_status.data == 0:
                        self.planner_status.data = 1
                        self.status_pub.publish(self.planner_status)

                    rospy.loginfo("Goal Arrived")
                if self.is_fear_act:
                    is_track_ahead = self.isForwardTraking(self.waypoints)
                    self.fearPathDetection(self.fear, is_track_ahead)
                    if self.is_fear_reaction:
                        rospy.logwarn_throttle(2.0, "current path prediction is invaild.")
                        # planner status -> Fails
                        if self.planner_status.data == 0:
                            self.planner_status.data = -1
                            self.status_pub.publish(self.planner_status)
                self.pubPath(self.waypoints, self.is_goal_init)
            r.sleep()
        rospy.spin()

    def pubPath(self, waypoints, is_goal_init=True):
        path = Path()
        fear_path = Path()
        if is_goal_init:
            for p in waypoints.squeeze(0):
                pose = PoseStamped()
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                path.poses.append(pose)
        # add header
        path.header.frame_id = fear_path.header.frame_id = self.frame_id
        path.header.stamp = fear_path.header.stamp = self.image_time
        # publish fear path
        if self.is_fear_reaction:
            fear_path.poses = copy.deepcopy(path.poses)
            path.poses = path.poses[:1]
        # publish path
        self.fear_path_pub.publish(fear_path)
        self.path_pub.publish(path)
        return

    def fearPathDetection(self, fear, is_forward):
        if fear > 0.5 and is_forward:
            if not self.is_fear_reaction:
                self.fear_buffter = self.fear_buffter + 1
        elif self.is_fear_reaction:
            self.fear_buffter = self.fear_buffter - 1
        if self.fear_buffter > self.buffer_size:
            self.is_fear_reaction = True
        elif self.fear_buffter <= 0:
            self.is_fear_reaction = False
        return None

    def isForwardTraking(self, waypoints):
        xhead = np.array([1.0, 0])
        phead = None
        for p in waypoints.squeeze(0):
            if torch.norm(p[0:2]).item() > self.track_dist:
                phead = np.array([p[0].item(), p[1].item()])
                phead /= np.linalg.norm(phead)
                break
        if np.all(phead != None) and phead.dot(xhead) > 1.0 - self.ang_thred:
            return True
        return False

    def joyCallback(self, joy_msg):
        if joy_msg.buttons[4] > 0.9:
            rospy.loginfo("Switch to Smart Joystick mode ...")
            self.is_smartjoy = True
            # reset fear reaction
            self.fear_buffter = 0
            self.is_fear_reaction = False
        if self.is_smartjoy:
            if np.sqrt(joy_msg.axes[3]**2 + joy_msg.axes[4]**2) < 1e-3:
                # reset fear reaction
                self.fear_buffter = 0
                self.is_fear_reaction = False
                self.ready_for_planning = False
                self.is_goal_init = False
            else:
                joy_goal = PointStamped()
                joy_goal.header.frame_id = self.frame_id
                joy_goal.point.x = joy_msg.axes[4] * self.joyGoal_scale
                joy_goal.point.y = joy_msg.axes[3] * self.joyGoal_scale
                joy_goal.point.z = 0.0
                joy_goal.header.stamp = rospy.Time.now()
                self.goal_pose = joy_goal
                self.is_goal_init = True
                self.is_goal_processed = False
        return

    def goalCallback(self, msg):
        rospy.loginfo("Recevied a new goal")
        self.goal_pose = msg
        self.is_smartjoy = False
        self.is_goal_init = True
        self.is_goal_processed = False
        # reset fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        # reste planner status
        self.planner_status.data = 0
        return

    def imageCallback(self, depth_msg: Image, sem_msg: Image):
        rospy.loginfo("Received depth image %s: %d"%(depth_msg.header.frame_id, depth_msg.header.seq))
        rospy.loginfo("Received sem image   %s: %d"%(sem_msg.header.frame_id, sem_msg.header.seq))
        self.image_time = depth_msg.header.stamp
        
        # convert depth image to numpy array
        frame = ros_numpy.numpify(depth_msg)
        frame[~np.isfinite(frame)] = 0
        if self.uint_type:
            frame = frame / 1000.0
        frame[frame > self.depth_max] = 0.0
        # DEBUG - Visual Image
        # img = PIL.Image.fromarray((frame * 255 / np.max(frame[frame>0])).astype('uint8'))
        # img.show()
        if self.image_flip:
            frame = PIL.Image.fromarray(frame)
            self.depth_img = np.array(frame.transpose(PIL.Image.ROTATE_180))
        else:
            self.depth_img = frame

        # convert rgb image to numpy array
        frame = ros_numpy.numpify(sem_msg)
        # DEBUG - Visual Image
        # img = PIL.Image.fromarray((frame)).astype('uint8'))
        # img.show()
        self.sem_img = frame
        
        # transform goal into robot frame
        if self.is_goal_init:
            goal_robot_frame = self.goal_pose;
            if not self.goal_pose.header.frame_id == self.frame_id:
                try:
                    goal_robot_frame.header.stamp = self.tf_listener.getLatestCommonTime(self.goal_pose.header.frame_id,
                                                                                         self.frame_id)
                    goal_robot_frame = self.tf_listener.transformPoint(self.frame_id, goal_robot_frame)
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr("Fail to transfer the goal into base frame.")
                    return
            goal_robot_frame = torch.tensor([goal_robot_frame.point.x, goal_robot_frame.point.y, goal_robot_frame.point.z], dtype=torch.float32)[None, ...]
            self.goal_rb = goal_robot_frame
        else:
            return
        
        # declare ready for planning
        self.ready_for_planning = True
        self.is_goal_processed  = True
        return

if __name__ == '__main__':

    node_name = "viplanner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument(
        'main_freq',       
        type=int,    
        default=5,                          
        help="frequency of path planner"
    )
    parser.add_argument(
        'uint_type',       
        type=bool,   
        default=False,                      
        help="image in uint type or not"
    )
    parser.add_argument(
        'robot_id',        
        type=str,    
        default='base',                     
        help='robot TF frame id'
    )
    parser.add_argument(
        'world_id',        
        type=str,    
        default='odom',                     
        help='world TF frame id'
    )
    parser.add_argument(
        'depth_max',       
        type=float,  
        default=10.0,                       
        help='max depth distance in image'
    )
    parser.add_argument(
        'image_flip',      
        type=bool,   
        default=True,                       
        help='is the image fliped'
    )
    parser.add_argument(
        'conv_dist',       
        type=float,  
        default=0.5,                        
        help='converge range to the goal'
    )
    parser.add_argument(
        'is_fear_act',     
        type=bool,   
        default=True,                       
        help='is open fear action or not'
    )
    parser.add_argument(
        'buffer_size',     
        type=int,    
        default=10,                         
        help='buffer size for fear reaction'
    )
    parser.add_argument(
        'angular_thred',   
        type=float,  
        default=0.3,                        
        help='angular thred for turning'
    )
    parser.add_argument(
        'track_dist',      
        type=float,  
        default=0.5,                        
        help='look ahead distance for path tracking'
    )
    parser.add_argument(
        'joyGoal_scale',   
        type=float,  
        default=0.5,                        
        help='distance for joystick goal'
    )
    
    # ROS topics
    parser.add_argument(
        'depth_topic',     
        type=str,    
        default='/rgbd_camera/depth/image', 
        help='depth image ros topic'
    )
    parser.add_argument(
        'goal_topic',      
        type=str,    
        default='/way_point',               
        help='goal waypoint ros topic'
    )
    parser.add_argument(
        'path_topic',      
        type=str,    
        default='/path',                    
        help='VIP Path topic'
    )
    parser.add_argument(
        'sem_topic',       
        type=str,    
        default='/m2f_sem',
        help='rgb camera topic'
    )

    # VIPlannerInferenceConfig
    parser.add_argument(
        'model_save',      
        type=str,    
        default='/models/plannernet.pt',    
        help="read model"
    )
    parser.add_argument(
        'crop_size',       
        type=tuple,  
        default=[360,640],                  
        help='image crop size'
    )
    parser.add_argument(
        'sensor_offset_x', 
        type=float,  
        default=0.0,                        
        help='sensor offset X'
    )
    parser.add_argument(
        'sensor_offset_y', 
        type=float,  
        default=0.0,                        
        help='sensor offset Y'
    )
    parser.add_argument(
        'semantics',       
        type=bool,   
        default=False,                      
        help='use semantics or not'
    )
    
    args = parser.parse_args()
    args.model_save = planner_path + args.model_save

    node = VIPlannerNode(args)

    node.spin()
