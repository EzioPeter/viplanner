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
import cv2
import copy
import time
import numpy as np
import scipy.spatial.transform as stf
from typing import Optional
import PIL

# ROS
import rospy
import rospkg
import tf
from std_msgs.msg import Float32, Int16, Header
from sensor_msgs.msg import Image, Joy, CameraInfo, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
import ros_numpy
import cv_bridge

import warnings
warnings.filterwarnings('ignore')

# init ros node
rospack = rospkg.RosPack()
pack_path = rospack.get_path('viplanner_node')
sys.path.append(pack_path)

# visual imperative planner
from model_src.vip_inference import VIPlannerInference
from model_src.m2f_inference import Mask2FormerInference
from utils.rosutil import ROSArgparse

# conversion matrix from ROS camera convention (z-forward, y-down, x-right) to robotics convention (x-forward, y-left, z-up)
ROS_TO_ROBOTICS_MAT = stf.Rotation.from_euler("XYZ", [-90, 0, -90], degrees=True).as_matrix()
CAMERA_FLIP_MAT     = stf.Rotation.from_euler("XYZ", [180, 0, 0],   degrees=True).as_matrix()


class VIPlannerNode:
    """VIPlanner ROS Node Class"""

    debug: bool = False

    def __init__(self, cfg):
        super(VIPlannerNode, self).__init__()
        self.cfg = cfg

        # init planner algo class
        self.vip_algo = VIPlannerInference(self.cfg)
        
        if self.vip_algo.train_cfg.sem:
            # init semantic network
            self.m2f_inference = Mask2FormerInference(
                config_file=args.m2f_config_path,
                model_weights=args.m2f_model_path,
            )
            self.m2f_timer_data = Float32()
            self.m2f_timer_pub  = rospy.Publisher(self.cfg.m2f_timer_topic, Float32, queue_size=10)

        # init transforms
        self.tf_listener = tf.TransformListener()
        
        # init bridge
        self.bridge = cv_bridge.CvBridge()
        
        # init flags
        self.is_goal_init               = False
        self.ready_for_planning_depth   = False
        self.ready_for_planning_rgb_sem = False
        self.is_goal_processed          = False
        self.is_smartjoy                = False
        self.goal_cam_frame_set         = False
        
        # planner status
        self.planner_status = Int16()
        self.planner_status.data = 0
        
        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        
        # process time
        self.vip_timer_data = Float32()
        self.vip_timer_pub  = rospy.Publisher('/vip_timer', Float32, queue_size=10)
        
        # depth and rgb image message
        self.depth_header: Header = Header()
        self.depth_header.stamp = rospy.get_rostime()
        self.depth_img: np.ndarray              = None
        self.depth_pose: np.ndarray             = None
        self.sem_rgb_img: np.ndarray            = None
        self.sem_rgb_odom: np.ndarray           = None
        self.pix_depth_cam_frame: np.ndarray    = None
        self.odom: torch.Tensor                 = None
        rospy.Subscriber(self.cfg.depth_topic, Image, callback=self.depthCallback, queue_size=1, buff_size=2**24)
        if self.cfg.compressed:
            rospy.Subscriber(self.cfg.rgb_topic, CompressedImage, callback=self.imageCallbackCompressed, queue_size=1, buff_size=2**24)
        else:
            rospy.Subscriber(self.cfg.rgb_topic, Image, callback=self.imageCallback, queue_size=1, buff_size=2**24)

        # subscribe to further topics
        rospy.Subscriber(self.cfg.goal_topic, PointStamped, self.goalCallback)
        rospy.Subscriber("/joy", Joy, self.joyCallback, queue_size=10)

        # camera info subscribers
        self.K_depth: np.ndarray = np.zeros((3,3))
        self.K_rgb: np.ndarray   = np.zeros((3,3))
        self.depth_intrinsics_init: bool = False
        self.rgb_intrinsics_init: bool = False
        rospy.Subscriber(self.cfg.depth_info_topic, CameraInfo, callback=self.depthCamInfoCallback)
        rospy.Subscriber(self.cfg.rgb_info_topic,   CameraInfo, callback=self.rgbCamInfoCallback)
        
        # planning status topics
        self.status_pub = rospy.Publisher('/vip_planner_status', Int16, queue_size=10)

        # path topics
        self.path_pub  = rospy.Publisher(self.cfg.path_topic, Path, queue_size=10)
        self.fear_path_pub = rospy.Publisher(self.cfg.path_topic + "_fear", Path, queue_size=10)

        # viz semantic image
        self.m2f_pub = rospy.Publisher("/m2f_sem_img", Image, queue_size=1)
         
        rospy.loginfo("VIPlanner Ready.")
        return

    def spin(self):
        r = rospy.Rate(self.cfg.main_freq)
        while not rospy.is_shutdown():
            if all((self.ready_for_planning_rgb_sem, self.ready_for_planning_depth, self.is_goal_init, self.goal_cam_frame_set)):
                # copy current data
                cur_rgb_image = self.sem_rgb_img.copy()
                cur_depth_image = self.depth_img.copy()
                cur_depth_pose = self.depth_pose.copy()
                cur_rgb_pose = self.sem_rgb_odom.copy()

                # warp rgb image
                start = time.time()
                if self.pix_depth_cam_frame is None:
                    self.initPixArray(cur_depth_image.shape)
                crop_image, overlap_ratio, depth_zero_ratio = self.imageWarp(cur_rgb_image, cur_depth_image, cur_rgb_pose, cur_depth_pose)
                time_warp = time.time() - start

                if overlap_ratio < self.cfg.overlap_ratio_thres:
                    rospy.logwarn_throttle(2.0, f"Waiting for new semantic image since overlap ratio is {overlap_ratio} < {self.cfg.overlap_ratio_thres}, whith depth zero ratio {depth_zero_ratio}")
                    self.pubPath(np.zeros((51, 3)), self.is_goal_init)
                    continue
                
                if depth_zero_ratio > self.cfg.depth_zero_ratio_thres:
                    rospy.logwarn_throttle(2.0, f"Waiting for new depth image since depth zero ratio is {depth_zero_ratio} > {self.cfg.depth_zero_ratio_thres}, whith overlap ratio {overlap_ratio}")
                    self.pubPath(np.zeros((51, 3)), self.is_goal_init)
                    continue
                
                # Network Planning
                start = time.time()
                waypoints, fear = self.vip_algo.plan(cur_depth_image, crop_image, self.goal_cam_frame)
                time_planner = time.time() - start
                
                start = time.time()
                
                # transform waypoint to robot frame (prev in depth cam frame with robotics convention)
                waypoints = (self.cam_rot @ waypoints.T).T + self.cam_offset
                
                # publish time
                self.vip_timer_data.data = time_planner * 1000
                self.vip_timer_pub.publish(self.vip_timer_data)
                
                # check goal less than converage range
                if (np.sqrt(self.goal_cam_frame[0][0]**2 + self.goal_cam_frame[0][1]**2) < self.cfg.conv_dist) and self.is_goal_processed and (not self.is_smartjoy):
                    self.ready_for_planning = False
                    self.is_goal_init = False
                    # planner status -> Success
                    if self.planner_status.data == 0:
                        self.planner_status.data = 1
                        self.status_pub.publish(self.planner_status)
                    rospy.loginfo("Goal Arrived")
                
                # check for path with high risk (=fear) path
                if self.cfg.is_fear_act:
                    is_track_ahead = self.isForwardTraking(waypoints)
                    self.fearPathDetection(fear, is_track_ahead)
                    if self.is_fear_reaction:
                        rospy.logwarn_throttle(2.0, "current path prediction is invaild.")
                        # planner status -> Fails
                        if self.planner_status.data == 0:
                            self.planner_status.data = -1
                            self.status_pub.publish(self.planner_status)
                
                # publish path
                self.pubPath(waypoints, self.is_goal_init)
                
                time_other = time.time() - start
                if self.vip_algo.train_cfg.pre_train_sem:
                    print(f"Path predicted in {round(time_warp + self.time_sem + time_planner + time_other, 4)}s \t warp: {round(time_warp, 4)}s \t sem: {round(self.time_sem, 4)}s \t planner: {round(time_planner, 4)}s \t other: {round(time_other, 4)}s")
                    self.time_sem = 0
                else:
                    print(f"Path predicted in {round(time_warp + time_planner + time_other, 4)}s \t warp: {round(time_warp, 4)}s \t planner: {round(time_planner, 4)}s \t other: {round(time_other, 4)}s")

            r.sleep()
        rospy.spin()

    """RGB/ SEM IMAGE WARP"""
    def initPixArray(self, img_shape: tuple):
        # get image plane mesh grid
        pix_u = np.arange(0, img_shape[1])
        pix_v = np.arange(0, img_shape[0])
        grid = np.meshgrid(pix_u, pix_v)
        pixels = np.vstack(list(map(np.ravel, grid))).T
        pixels = np.hstack(
            [pixels, np.ones((len(pixels), 1))]
        )  # add ones for 3D coordinates           
        
        # transform to camera frame
        k_inv = np.linalg.inv(self.K_depth)
        pix_cam_frame = np.matmul(k_inv, pixels.T)  # pixels in ROS camera convention (z forward, x right, y down)
        
        # reorder to be in "robotics" axis order (x forward, y left, z up)
        self.pix_depth_cam_frame = pix_cam_frame[[2, 0, 1], :].T  * np.array([1, -1, -1])
        return
        
    def imageWarp(self, rgb_img: np.ndarray, depth_img: np.ndarray, pose_rgb: np.ndarray, pose_depth: np.ndarray) -> np.ndarray:
        # get 3D points of depth image
        depth_rot = stf.Rotation.from_quat(pose_depth[3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT # convert orientation from ROS camera to robotics=world frame
        if not self.cfg.image_flip: # rotation is included in ROS_TO_ROBOTICS_MAT and has to be removed when not fliped
            depth_rot = depth_rot @ CAMERA_FLIP_MAT
        dep_im_reshaped = depth_img.reshape(-1, 1)
        depth_zero_ratio = np.sum(np.round(dep_im_reshaped, 5) == 0) / len(dep_im_reshaped)
        points = dep_im_reshaped * (depth_rot @ self.pix_depth_cam_frame.T).T + pose_depth[:3]
        
        # transform points to semantic camera frame
        points_sem_cam_frame = ((stf.Rotation.from_quat(pose_rgb[3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT @ CAMERA_FLIP_MAT).T @ (points - pose_rgb[:3]).T).T
        
        # normalize points
        points_sem_cam_frame_norm = points_sem_cam_frame / points_sem_cam_frame[:, 0][:, np.newaxis]
       
        # reorder points be camera convention (z-forward)
        points_sem_cam_frame_norm = points_sem_cam_frame_norm[:, [1, 2, 0]]  * np.array([-1, -1, 1])
        # transform points to pixel coordinates
        pixels = (self.K_rgb @ points_sem_cam_frame_norm.T).T
        # filter points outside of image
        filter_idx = (pixels[:, 0] >= 0) & (pixels[:, 0] < rgb_img.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < rgb_img.shape[0])
        # get semantic annotation
        rgb_pixels = np.zeros((pixels.shape[0], 3))
        rgb_pixels[filter_idx] = rgb_img[pixels[filter_idx, 1].astype(int)-1, pixels[filter_idx, 0].astype(int)-1]
        rgb_warped = rgb_pixels.reshape(depth_img.shape[0], depth_img.shape[1], 3)
        # overlap ratio
        overlap_ratio = np.sum(filter_idx) / pixels.shape[0]
        
        # DEBUG
        if self.debug:
            print("depth_rot", stf.Rotation.from_matrix(depth_rot).as_euler("xyz", degrees=True))
            rgb_rot = stf.Rotation.from_quat(pose_rgb[3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT @ CAMERA_FLIP_MAT
            print("rgb_rot", stf.Rotation.from_matrix(rgb_rot).as_euler("xyz", degrees=True))       
            
            import matplotlib.pyplot as plt
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
            ax1.imshow(depth_img)
            ax2.imshow(rgb_img)
            ax3.imshow(rgb_warped / 255)
            ax4.imshow(depth_img)
            ax4.imshow(rgb_warped / 255, alpha=0.5)
            plt.savefig(os.path.join(os.getcwd(), "depth_sem_warp.png"))
            # plt.show()  
            plt.close()  
        
        # reshape to image
        return rgb_warped, overlap_ratio, depth_zero_ratio
    
    """PATH PUB, GOLA SUB and FEAR DETECTION"""

    def pubPath(self, waypoints, is_goal_init=True):
        path = Path()
        fear_path = Path()
        if is_goal_init:
            for p in waypoints:
                # gte individual pose in depth frame
                pose = PoseStamped()
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                # append to path
                path.poses.append(pose)
        # add header
        path.header.frame_id = fear_path.header.frame_id = self.cfg.robot_id
        path.header.stamp = fear_path.header.stamp = self.depth_header.stamp
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
        for p in waypoints:
            if np.linalg.norm(p[0:2]) > self.cfg.track_dist:
                phead = p[0:2] / np.linalg.norm(p[0:2])
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

    """RGB IMAGE AND DEPTH CALLBACKS"""
    def poseCallback(self, frame_id: str, target_frame_id: Optional[str] = None):
        target_frame_id = target_frame_id if target_frame_id else self.cfg.world_id
        try:            
            self.tf_listener.waitForTransform(target_frame_id, frame_id, rospy.Time(0), rospy.Duration(4.0))
            t = self.tf_listener.getLatestCommonTime(target_frame_id, frame_id)
            pose = self.tf_listener.lookupTransform(target_frame_id, frame_id, t)
            pose = np.hstack(pose)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr(f"Fail to transfer {frame_id} into {target_frame_id} frame.")
        return pose
            
    def imageCallback(self, rgb_msg: Image):
        rospy.logdebug("Received rgb image %s: %d"%(rgb_msg.header.frame_id,   rgb_msg.header.seq))

        # image pose
        pose = self.poseCallback(rgb_msg.header.frame_id)
        
        # RGB image
        try:
            image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            print(e)

        if self.vip_algo.train_cfg.sem:
            image = self.semPrediction(image)
        
        self.sem_rgb_odom = pose
        self.sem_rgb_img = image
        return        
            
    def imageCallbackCompressed(self, rgb_msg: CompressedImage):
        rospy.logdebug(f"Received rgb   image {rgb_msg.header.frame_id}: {rgb_msg.header.stamp.to_sec()}")
        # image pose
        pose = self.poseCallback(rgb_msg.header.frame_id)
        
        # RGB Image
        try:
            rgb_arr = np.frombuffer(rgb_msg.data, np.uint8)
            image = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
        except cv_bridge.CvBridgeError as e:
            print(e)

        if self.vip_algo.train_cfg.sem:
            image = self.semPrediction(image)
        
        self.sem_rgb_img = image
        self.sem_rgb_odom = pose
        self.sem_rgb_new = True
        self.ready_for_planning_rgb_sem = True
        
        # publish the image
        if self.vip_algo.train_cfg.sem:
            image = cv2.resize(image, (480, 360))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            sem_msg = ros_numpy.msgify(Image, image, encoding="8UC3")
            sem_msg.header.stamp = rospy.Time.now()
            self.m2f_pub.publish(sem_msg)
        return
    
    def semPrediction(self, image):
        # semantic estimation with image in BGR format
        start = time.time()
        image = self.m2f_inference.predict(image)
        self.time_sem = time.time() - start
        # publish prediction time
        self.m2f_timer_data.data = self.time_sem * 1000
        self.m2f_timer_pub.publish(self.m2f_timer_data)
        return image
        
    def depthCallback(self, depth_msg: Image):
        rospy.logdebug(f"Received depth image {depth_msg.header.frame_id}: {depth_msg.header.stamp.to_sec()}")

        # image time and pose
        self.depth_header = depth_msg.header
        self.depth_pose = self.poseCallback(depth_msg.header.frame_id)
        # DEPTH Image
        image = ros_numpy.numpify(depth_msg)
        image[~np.isfinite(image)] = 0
        if self.cfg.depth_uint_type:
            image = image / 1000.0
        image[image > self.cfg.max_depth] = 0.0
        if self.cfg.image_flip:
            image = PIL.Image.fromarray(image)
            self.depth_img = np.array(image.transpose(PIL.Image.Transpose.ROTATE_180))
        else:
            self.depth_img = image
        
        # transform goal into robot frame
        if self.is_goal_init:
            goal_robot_frame = self.goal_pose;
            if not self.goal_pose.header.frame_id == self.cfg.robot_id:
                try:
                    goal_robot_frame.header.stamp = self.tf_listener.getLatestCommonTime(self.goal_pose.header.frame_id,
                                                                                         self.cfg.robot_id)
                    goal_robot_frame = self.tf_listener.transformPoint(self.cfg.robot_id, goal_robot_frame)
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr(f"Goal: Fail to transfer {self.goal_pose.header.frame_id} into {self.cfg.robot_id}")
                    return
            # get transform from robot frame to depth camera frame
            tf_robot_depth = self.poseCallback(depth_msg.header.frame_id, self.cfg.robot_id)
            self.cam_offset = tf_robot_depth[0:3]
            self.cam_rot = stf.Rotation.from_quat(tf_robot_depth[3:7]).as_matrix() @ ROS_TO_ROBOTICS_MAT
            if not self.cfg.image_flip:  # rotation is included in ROS_TO_ROBOTICS_MAT and has to be removed when not fliped
                self.cam_rot = self.cam_rot @ CAMERA_FLIP_MAT
            goal_robot_frame = np.array([goal_robot_frame.point.x, goal_robot_frame.point.y, goal_robot_frame.point.z])
            goal_cam_frame = self.cam_rot.T @ (goal_robot_frame - self.cam_offset).T
            self.goal_cam_frame = torch.tensor(goal_cam_frame, dtype=torch.float32)[None, ...]
            self.goal_cam_frame_set = True

            if self.debug:
                print("CAM ROT", stf.Rotation.from_matrix(self.cam_rot).as_euler("xyz", degrees=True)) 
                       
        # declare ready for planning
        self.ready_for_planning_depth = True
        self.is_goal_processed  = True
        return
    
    """ Camera Info Callbacks"""
    def depthCamInfoCallback(self, cam_info_msg: CameraInfo):
        if not self.depth_intrinsics_init:
            rospy.loginfo("Received depth camera info")
            self.K_depth = cam_info_msg.K
            self.K_depth = np.array(self.K_depth).reshape(3, 3)
            self.depth_intrinsics_init = True
        return

    def rgbCamInfoCallback(self, cam_info_msg: CameraInfo):
        if not self.rgb_intrinsics_init:
            rospy.loginfo("Received rgb camera info")
            self.K_rgb = cam_info_msg.K
            self.K_rgb = np.array(self.K_rgb).reshape(3, 3)
            self.rgb_intrinsics_init = True
        return

 
if __name__ == '__main__':

    node_name = "viplanner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    # planning
    parser.add_argument('main_freq',        type=int,   default=5,                          
        help="frequency of path planner")
    parser.add_argument('image_flip',       type=bool,  default=True,                       
        help='is the image fliped'
    )
    parser.add_argument('conv_dist',        type=float, default=0.5,                        
        help='converge range to the goal'
    )
    parser.add_argument('max_depth',        type=float, default=10.0,                       
        help='max depth distance in image'
    )
    parser.add_argument('overlap_ratio_thres',type=float, default=0.7,                       
        help='overlap threshold betweens sem/rgb and depth image'
    )
    parser.add_argument('depth_zero_ratio_thres',type=float, default=0.7,                       
        help='ratio of depth image that is non-zero'
    )
    # networks 
    parser.add_argument('model_save',       type=str,   default='models/vip_models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD',    
        help="model directory (within should be a file called model.pt and model.yaml)"
    )
    parser.add_argument('m2f_cfg_file',     type=str,   default='models/coco_panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml',   
        help="config file for m2f model (or pre-trained backbone for direct RGB input)"
    )
    parser.add_argument('m2f_model_path',   type=str,   default='models/coco_panoptic/swin/model_final_9fd0ae.pkl',   
        help="read model for m2f model (or pre-trained backbone for direct RGB input)"
    )
    # ROS topics
    parser.add_argument('depth_topic',     type=str,    default='/rgbd_camera/depth/image', 
        help='depth image ros topic'
    )
    parser.add_argument('depth_info_topic',type=str,    default='/depth_camera_front_upper/depth/camera_info', 
        help='depth image info topic (get intrinsic matrix)'
    )
    parser.add_argument('rgb_topic',       type=str,    default='/wide_angle_camera_front/image_raw/compressed',
        help='rgb camera topic'
    )
    parser.add_argument('rgb_info_topic',  type=str,    default='/wide_angle_camera_front/camera_info',
        help='rgb camera info topic (get intrinsic matrix)'
    )
    parser.add_argument('goal_topic',      type=str,    default='/way_point',               
        help='goal waypoint ros topic'
    )
    parser.add_argument('path_topic',      type=str,    default='/path',                    
        help='VIP Path topic'
    )
    parser.add_argument('m2f_timer_topic', type=str,    default='/m2f_timer', 
        help='Time needed for semantic segmentation'
    )
    parser.add_argument('depth_uint_type', type=bool,   default=False,                      
        help="image in uint type or not"
    )
    parser.add_argument('compressed',      type=bool,   default=True, 
        help='If compressed rgb topic is used'
    )

    # frame_ids
    parser.add_argument('robot_id',        type=str,     default='base',                     
        help='robot TF frame id'
    )
    parser.add_argument('world_id',        type=str,     default='odom',                     
        help='world TF frame id'
    )

    # fear reaction
    parser.add_argument('is_fear_act',     type=bool,    default=True,                       
        help='is open fear action or not'
    )
    parser.add_argument('buffer_size',     type=int,     default=10,                         
        help='buffer size for fear reaction'
    )
    parser.add_argument('angular_thred',   type=float,   default=0.3,                        
        help='angular thred for turning'
    )
    parser.add_argument('track_dist',      type=float,   default=0.5,                        
        help='look ahead distance for path tracking'
    )
    # smart joystick
    parser.add_argument('joyGoal_scale',   type=float,   default=0.5,                        
        help='distance for joystick goal'
    )

    args = parser.parse_args()

    # model save path
    args.model_save      = os.path.join(pack_path, args.model_save)
    args.m2f_config_path = os.path.join(pack_path, args.m2f_cfg_file)
    args.m2f_model_path  = os.path.join(pack_path, args.m2f_model_path)
    
    node = VIPlannerNode(args)

    node.spin()
