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
from typing import Optional, Union
import PIL

# ROS
import rospy
import rospkg
import tf
from std_msgs.msg import Float32, Int16
from sensor_msgs.msg import Image, Joy, CameraInfo, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
import ros_numpy
import message_filters
import cv_bridge

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
CAMERA_FLIP_MAT = stf.Rotation.from_euler("XYZ", [180, 0, 0], degrees=True).as_matrix()


class VIPlannerNode:
    """VIPlanner ROS Node Class"""

    debug: bool = False

    def __init__(self, cfg):
        super(VIPlannerNode, self).__init__()
        self.cfg = cfg

        # init planner algo class
        self.vip_algo = VIPlannerInference(
            model_save=self.cfg.model_save,
            sensor_offset_x=self.cfg.sensor_offset_x,
            sensor_offset_y=self.cfg.sensor_offset_y,
        )
        # init semantic network
        self.m2f_inference = Mask2FormerInference(
            config_file=args.m2f_config_path,
            model_weights=args.m2f_model_path,
        )

        # init transforms        
        self.tf_listener = tf.TransformListener()
        # init bridge
        self.bridge = cv_bridge.CvBridge()
        # init flags
        self.image_time = rospy.get_rostime()
        self.is_goal_init = False
        self.ready_for_planning = False
        self.is_goal_processed = False
        self.is_smartjoy = False

        # planner status
        self.planner_status = Int16()
        self.planner_status.data = 0


        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        
        # process time
        self.vip_timer_data = Float32()
        self.m2f_timer_data = Float32()
        self.vip_timer_pub  = rospy.Publisher('/vip_timer', Float32, queue_size=10)
        self.m2f_timer_pub  = rospy.Publisher(self.cfg.m2f_timer_topic, Float32, queue_size=10)
        
        # depth and rgb image message --> time syncronization by message_filters
        self.depth_img: np.ndarray = None
        self.rgb_img: np.ndarray = None
        self.pix_depth_cam_frame: np.ndarray = None
        self.odom: torch.Tensor = None
        img_depth_sub = message_filters.Subscriber(self.cfg.depth_topic, Image)
        if self.cfg.compressed:
            img_rgb_sub = message_filters.Subscriber(self.cfg.rgb_topic, CompressedImage)
        else:
            img_rgb_sub = message_filters.Subscriber(self.cfg.rgb_topic, Image)
        ts = message_filters.ApproximateTimeSynchronizer([img_depth_sub, img_rgb_sub], 40, 0.05)
        if self.cfg.compressed:
            ts.registerCallback(self.imageCallbackCompressed)
        else:
            ts.registerCallback(self.imageCallback)
        
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

        # path visualization topics
        if self.cfg.path_viz:
            self.img_pub_dep = rospy.Publisher(self.cfg.viz_path_depth_topic, Image, queue_size=10)
            self.img_pub_sem = rospy.Publisher(self.cfg.viz_path_sem_topic, Image, queue_size=10)
            self.traj_viz: Optional[TrajViz] = None
        rospy.loginfo("VIPlanner Ready.")
        return

    def spin(self):
        r = rospy.Rate(self.cfg.main_freq)
        while not rospy.is_shutdown():
            if self.ready_for_planning and self.is_goal_init:
                # main planning starts
                cur_rgb_image = self.rgb_img.copy()
                cur_depth_image = self.depth_img.copy()
                cur_depth_pose = self.depth_pose.copy()
                cur_rgb_pose = self.rgb_pose.copy()

                # warp rgb image
                start = time.time()
                if self.pix_depth_cam_frame is None:
                    self.initPixArray(cur_depth_image.shape)
                crop_rgb_image = self.imageWarp(cur_rgb_image, cur_depth_image, cur_rgb_pose, cur_depth_pose)
                time_warp = time.time() - start

                # semantic estimation
                start = time.time()
                cur_sem_image = self.m2f_inference.predict(crop_rgb_image)
                time_sem = time.time() - start

                # Network Planning
                start = time.time()
                self.preds, self.waypoints, self.fear = self.vip_algo.plan(cur_depth_image, cur_sem_image, self.goal_rb)
                time_planner = time.time() - start
                
                start = time.time()
                
                # publish time
                self.m2f_timer_data.data = time_sem * 1000
                self.m2f_timer_pub.publish(self.m2f_timer_data)
                self.vip_timer_data.data = time_planner * 1000
                self.vip_timer_pub.publish(self.vip_timer_data)
                
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
                
                time_other = time.time() - start
                print(f"Path predicted in {round(time_warp + time_sem + time_planner + time_other, 4)}s \t warp: {round(time_warp, 4)}s \t sem: {round(time_sem, 4)}s \t planner: {round(time_planner, 4)}s \t other: {round(time_other, 4)}s")

                # vizualize path
                if self.cfg.path_viz:
                    self.pubRenderImage(self.preds, self.waypoints, self.odom, self.goal_rb, self.fear, cur_depth_image, cur_sem_image)

            r.sleep()
        rospy.spin()

    """RGB IMAGE WARP"""
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
        dep_im_reshaped = depth_img.reshape(-1, 1)
        points = dep_im_reshaped * (depth_rot @ self.pix_depth_cam_frame.T).T + pose_depth[:3]
        
        # transform points to semantic camera frame
        if self.cfg.image_flip:
            points_sem_cam_frame = ((stf.Rotation.from_quat(pose_rgb[3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT @ CAMERA_FLIP_MAT).T @ (points - pose_rgb[:3]).T).T
        else:
            points_sem_cam_frame = ((stf.Rotation.from_quat(pose_rgb[3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT).T @ (points - pose_rgb[:3]).T).T
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
        
        # DEBUG
        if self.debug:
            import matplotlib.pyplot as plt
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
            ax1.imshow(depth_img)
            ax2.imshow(rgb_img)
            ax3.imshow(rgb_warped / 255)
            ax4.imshow(depth_img)
            ax4.imshow(rgb_warped / 255, alpha=0.5)
            plt.show()    
        
        # reshape to image
        return rgb_warped
    
    """PATH PUB, GOLA SUB and FEAR DETECTION"""

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
    
    """IMAGE CALLBACKS"""

    def imageCallback(self, depth_msg: Image, rgb_msg: Image):
        rospy.logdebug("Received rgb   image %s: %d"%(rgb_msg.header.frame_id,   rgb_msg.header.seq))
        rospy.logdebug("Received depth image %s: %d"%(depth_msg.header.frame_id, depth_msg.header.seq))        

        # image time
        self.image_time = depth_msg.header.stamp
        
        # RGB image
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            if not self.cfg.image_flip:
                # rotate image 90 degrees coutner clockwise
                self.rgb_img = cv2.rotate(rgb_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except cv_bridge.CvBridgeError as e:
            print(e)

        self.depthCallback(depth_msg)
        self.poseCallback(rgb_msg, depth_msg)
        return        
            
    def imageCallbackCompressed(self, depth_msg: Image, rgb_msg: CompressedImage):
        rospy.logdebug("Received rgb   image %s: %d"%(rgb_msg.header.frame_id,   rgb_msg.header.seq))
        rospy.logdebug("Received depth image %s: %d"%(depth_msg.header.frame_id, depth_msg.header.seq))

        # image time
        self.image_time = depth_msg.header.stamp

        # RGB Image
        try:
            rgb_arr = np.frombuffer(rgb_msg.data, np.uint8)
            self.rgb_img = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
            # rotate image 90 degrees coutner clockwise
            if not self.cfg.image_flip:
                self.rgb_img = cv2.rotate(self.rgb_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except cv_bridge.CvBridgeError as e:
            print(e)

        self.depthCallback(depth_msg)
        self.poseCallback(rgb_msg, depth_msg)
        return
    
    def depthCallback(self, depth_msg: Image):
        # DEPTH Image
        frame = ros_numpy.numpify(depth_msg)
        frame[~np.isfinite(frame)] = 0
        if self.cfg.depth_uint_type:
            frame = frame / 1000.0
        frame[frame > self.cfg.max_depth] = 0.0
        if self.cfg.image_flip:
            frame = PIL.Image.fromarray(frame)
            self.depth_img = np.array(frame.transpose(PIL.Image.Transpose.ROTATE_180))
        else:
            self.depth_img = frame
        return
    
    def poseCallback(self, rgb_msg: Union[Image, CompressedImage], depth_msg: Image):
        # get current pose of semantic and depth image
        try:            
            pose = self.tf_listener.lookupTransform(self.cfg.robot_id, rgb_msg.header.frame_id, rgb_msg.header.stamp)
            self.rgb_pose = np.hstack(pose)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr(f"RGB Image: Fail to transfer {rgb_msg.header.frame_id} into {self.cfg.robot_id} frame.")
        try:
            pose = self.tf_listener.lookupTransform(self.cfg.robot_id, depth_msg.header.frame_id, depth_msg.header.stamp)
            self.depth_pose = np.hstack(pose)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr(f"Depth Image: Fail to transfer {depth_msg.header.frame_id} into {self.cfg.robot_id} frame.")
        
       # get odom from TF for camera image visualization 
        try:
            self.tf_listener.waitForTransform(self.cfg.world_id, self.cfg.robot_id, rospy.Time(0), rospy.Duration(4.0))
            t = self.tf_listener.getLatestCommonTime(self.cfg.world_id, self.cfg.robot_id)
            (odom, ori) = self.tf_listener.lookupTransform(self.cfg.world_id, self.cfg.robot_id, t)
            odom.extend(ori)
            self.odom = torch.tensor(odom, dtype=torch.float32).unsqueeze(0)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr(f"Odom: Fail to transfer {self.cfg.world_id,} into {self.cfg.robot_id}")
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
                    rospy.logerr(f"Goal: Fail to transfer {self.goal_pose.header.frame_id} into {self.cfg.robot_id}")
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
    # networks 
    parser.add_argument('model_save',       type=str,   default='models/vip_models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD',    
        help="model directory (within should be a file called model.pt and model.yaml)"
    )
    parser.add_argument('m2f_cfg_file',     type=str,   default='models/coco_panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml',   
        help="config file for m2f model"
    )
    parser.add_argument('m2f_model_path',   type=str,   default='models/coco_panoptic/swin/model_final_9fd0ae.pkl',   
        help="read model"
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
    # sensor offset
    parser.add_argument('sensor_offset_x', type=float,   default=0.0,                        
        help='sensor offset X'
    )
    parser.add_argument('sensor_offset_y', type=float,   default=0.0,                        
        help='sensor offset Y'
    )
    # smart joystick
    parser.add_argument('joyGoal_scale',   type=float,   default=0.5,                        
        help='distance for joystick goal'
    )
    # Visualization config
    parser.add_argument('path_viz',         type=bool,   default=False,
        help='Publish TrajViz images to path evaluation'
    )
    parser.add_argument('viz_path_depth_topic',     type=str,     default='/viz_path_depth', 
        help='publish path projected in depth image'
    )
    parser.add_argument('viz_path_sem_topic',     type=str,     default='/viz_path_sem', 
        help='publish path projected in semantic image'
    )

    args = parser.parse_args()

    # import only if needed (currently error on Jetson)
    if args.path_viz:
        from model_src.viplanner.traj_cost_opt.traj_viz import TrajViz

    # model save path
    args.model_save      = os.path.join(pack_path, args.model_save)
    args.m2f_config_path = os.path.join(pack_path, args.m2f_cfg_file)
    args.m2f_model_path  = os.path.join(pack_path, args.m2f_model_path)
    
    node = VIPlannerNode(args)

    node.spin()
