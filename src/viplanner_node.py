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
from typing import Optional

# ROS
import rospy
import rospkg
import tf
from std_msgs.msg import Float32, Int16
from sensor_msgs.msg import Image, Joy, CameraInfo
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
import ros_numpy
import message_filters

# init ros node
rospack = rospkg.RosPack()
pack_path = rospack.get_path('viplanner_node')
sys.path.append(pack_path)

# visual imperative planner
from model_src.vip_inference import VIPlannerInference
from utils.rosutil import ROSArgparse


class VIPlannerNode:
    """VIPlanner ROS Node Class"""
    def __init__(self, cfg):
        super(VIPlannerNode, self).__init__()
        self.cfg = cfg

        # init planner algo class
        self.vip_algo = VIPlannerInference(
            model_save=self.cfg.model_save,
            sensor_offset_x=self.cfg.sensor_offset_x,
            sensor_offset_y=self.cfg.sensor_offset_y,
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
        self.depth_img: np.ndarray = None
        self.sem_img: np.ndarray = None
        self.odom: torch.Tensor = None
        img_depth_sub = message_filters.Subscriber(self.cfg.depth_topic, Image)
        img_sem_sub = message_filters.Subscriber(self.cfg.sem_topic, Image)
        ts = message_filters.TimeSynchronizer([img_depth_sub, img_sem_sub], 10)
        ts.registerCallback(self.imageCallback)
        
        # subscribe to further topics
        rospy.Subscriber(self.cfg.goal_topic, PointStamped, self.goalCallback)
        rospy.Subscriber("/joy", Joy, self.joyCallback, queue_size=10)

        # camera info subscribers
        self.K_depth: np.ndarray = None
        self.intrinsics_init: bool = False
        rospy.Subscriber(self.cfg.depth_cam_info_topic, CameraInfo, callback=self.depthCamInfoCallback)
        
        timer_topic = '/vip_timer'
        status_topic = '/vip_planner_status'
        
        # planning status topics
        self.timer_pub = rospy.Publisher(timer_topic, Float32, queue_size=10)
        self.status_pub = rospy.Publisher(status_topic, Int16, queue_size=10)

        # path topics
        self.path_pub  = rospy.Publisher(self.cfg.path_topic, Path, queue_size=10)
        self.fear_path_pub = rospy.Publisher(self.cfg.path_topic + "_fear", Path, queue_size=10)

        # path visualization topics
        if self.cfg.path_viz:
            self.img_pub_dep = rospy.Publisher(self.cfg.viz_path_depth_topic, Image, queue_size=10)
            self.img_pub_sem = rospy.Publisher(self.cfg.viz_path_sem_topic, Image, queue_size=10)
            self.traj_viz: Optional[TrajViz] = None
        rospy.loginfo("VIPlanner Ready.")

    def spin(self):
        r = rospy.Rate(self.cfg.main_freq)
        while not rospy.is_shutdown():
            if self.ready_for_planning and self.is_goal_init:
                # main planning starts
                cur_depth_image = self.depth_img.copy()
                cur_sem_image = self.sem_img.copy()
                start = time.time()
                # Network Planning
                self.preds, self.waypoints, self.fear = self.vip_algo.plan(cur_depth_image, cur_sem_image, self.goal_rb)
                end = time.time()
                self.timer_data.data = (end - start) * 1000
                self.timer_pub.publish(self.timer_data)
                print(self.preds)
                
                # check goal less than converage range
                if (np.sqrt(self.goal_rb[0][0]**2 + self.goal_rb[0][1]**2) < self.cfg.conv_dist) and self.is_goal_processed and (not self.is_smartjoy):
                    self.ready_for_planning = False
                    self.is_goal_init = False
                    # planner status -> Success
                    if self.planner_status.data == 0:
                        self.planner_status.data = 1
                        self.status_pub.publish(self.planner_status)
                    rospy.loginfo("Goal Arrived")
                
                # check for path with high risk (=fear) path
                if self.cfg.is_fear_act:
                    is_track_ahead = self.isForwardTraking(self.waypoints)
                    self.fearPathDetection(self.fear, is_track_ahead)
                    if self.is_fear_reaction:
                        rospy.logwarn_throttle(2.0, "current path prediction is invaild.")
                        # planner status -> Fails
                        if self.planner_status.data == 0:
                            self.planner_status.data = -1
                            self.status_pub.publish(self.planner_status)
                
                # publish path
                self.pubPath(self.waypoints, self.is_goal_init)
                
                # vizualize path
                if self.cfg.path_viz:
                    self.pubRenderImage(self.preds, self.waypoints, self.odom, self.goal_rb, self.fear, cur_depth_image, cur_sem_image)

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
        path.header.frame_id = fear_path.header.frame_id = self.cfg.robot_id
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
        if self.fear_buffter > self.cfg.buffer_size:
            self.is_fear_reaction = True
        elif self.fear_buffter <= 0:
            self.is_fear_reaction = False
        return None

    def isForwardTraking(self, waypoints):
        xhead = np.array([1.0, 0])
        phead = None
        for p in waypoints.squeeze(0):
            if torch.norm(p[0:2]).item() > self.cfg.track_dist:
                phead = np.array([p[0].item(), p[1].item()])
                phead /= np.linalg.norm(phead)
                break
        if np.all(phead != None) and phead.dot(xhead) > 1.0 - self.cfg.angular_thred:
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
                joy_goal.header.frame_id = self.cfg.robot_id
                joy_goal.point.x = joy_msg.axes[4] * self.cfg.joyGoal_scale
                joy_goal.point.y = joy_msg.axes[3] * self.cfg.joyGoal_scale
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

    def pubRenderImage(self, preds, waypoints, odom, goal, fear, dep_image, sem_image):
        if self.traj_viz is None:
            self.traj_viz = TrajViz(intrinsics=self.K_depth)

        if torch.cuda.is_available():
            odom = odom.cuda()
            goal = goal.cuda()
            
        image = self.traj_viz.VizImages(preds, waypoints, odom, goal, fear, dep_image)[0]
        ros_img = ros_numpy.msgify(Image, image, encoding='rgb8')
        self.img_pub_dep.publish(ros_img)
        return None
    
    def imageCallback(self, depth_msg: Image, sem_msg: Image):
        rospy.loginfo("Received depth image %s: %d"%(depth_msg.header.frame_id, depth_msg.header.seq))
        rospy.loginfo("Received sem image   %s: %d"%(sem_msg.header.frame_id, sem_msg.header.seq))
        self.image_time = depth_msg.header.stamp
        
        # convert depth image to numpy array
        frame = ros_numpy.numpify(depth_msg)
        frame[~np.isfinite(frame)] = 0
        if self.cfg.depth_uint_type:
            frame = frame / 1000.0
        frame[frame > self.cfg.depth_max] = 0.0
        if self.cfg.image_flip:
            frame = PIL.Image.fromarray(frame)
            self.depth_img = np.array(frame.transpose(PIL.Image.ROTATE_180))
        else:
            self.depth_img = frame

        # convert rgb image to numpy array
        frame = ros_numpy.numpify(sem_msg)
        self.sem_img = frame
        
       # get odom from TF for camera image visualization 
        try:
            self.tf_listener.waitForTransform(self.cfg.world_id, self.cfg.robot_id, rospy.Time(0), rospy.Duration(4.0))
            t = self.tf_listener.getLatestCommonTime(self.cfg.world_id, self.cfg.robot_id)
            (odom, ori) = self.tf_listener.lookupTransform(self.cfg.world_id, self.cfg.robot_id, t)
            odom.extend(ori)
            self.odom = torch.tensor(odom, dtype=torch.float32).unsqueeze(0)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Fail to get odomemrty from tf.")
            return
        
        # transform goal into robot frame
        if self.is_goal_init:
            goal_robot_frame = self.goal_pose;
            if not self.goal_pose.header.frame_id == self.cfg.robot_id:
                try:
                    goal_robot_frame.header.stamp = self.tf_listener.getLatestCommonTime(self.goal_pose.header.frame_id,
                                                                                         self.cfg.robot_id)
                    goal_robot_frame = self.tf_listener.transformPoint(self.cfg.robot_id, goal_robot_frame)
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

    def depthCamInfoCallback(self, cam_info_msg: CameraInfo):
        if not self.intrinsics_init:
            rospy.loginfo("Received depth camera info")
            self.K_depth = cam_info_msg.K
            self.K_depth = np.array(self.K_depth).reshape(3, 3)
            self.intrinsics_init = True
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
        'depth_uint_type',       
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
    parser.add_argument(
        'depth_cam_info_topic',     
        type=str,    
        default='/depth_camera_front_upper/depth/camera_info', 
        help='depth image info topic (get intrinsic matrix)'
    )
    parser.add_argument(
        'viz_path_depth_topic',     
        type=str,    
        default='/viz_path_depth', 
        help='publish path projected in depth image'
    )
    parser.add_argument(
        'viz_path_sem_topic',     
        type=str,    
        default='/viz_path_sem', 
        help='publish path projected in semantic image'
    )
    
    # VIPlannerInferenceConfig
    parser.add_argument(
        'model_save',      
        type=str,    
        default='models/vip_models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD',    
        help="model directory (within should be a file called model.pt and model.yaml)"
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
    
    # Visualization config
    parser.add_argument(
        'path_viz',
        type=bool,
        default=False,
        help='Publish TrajViz images to path evaluation'
    )
    
    args = parser.parse_args()
    
    if args.path_viz:
        from model_src.viplanner.traj_cost_opt.traj_viz import TrajViz

    # model save path
    args.model_save = os.path.join(pack_path, args.model_save)
    
    node = VIPlannerNode(args)

    node.spin()
