#!/usr/bin/env python3
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch
@author     Fan Yang
@email      fanyang1@ethz.ch


@brief      Mask2Former (M2F) ROS Node
"""

# python
import os
import sys
import time
import numpy as np
import PIL
import cv2 
from typing import Union
import scipy.spatial.transform as stf

# ROS
import rospy
import rospkg
import tf
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters
import ros_numpy
import cv_bridge

# init ros node
rospack = rospkg.RosPack()
pack_path = rospack.get_path('viplanner_node')
sys.path.append(pack_path)

# visual imperative planner
from model_src.m2f_inference import Mask2FormerInference
from utils.rosutil import ROSArgparse

# conversion matrix from ROS camera convention (z-forward, y-down, x-right) to robotics convention (x-forward, y-left, z-up)
ROS_TO_ROBOTICS_MAT = stf.Rotation.from_euler("XYZ", [-90, 0, -90], degrees=True).as_matrix()

class VIPlannerNode:
    """VIPlanner ROS Node Class"""
    
    debug: bool = False
    
    def __init__(self, args):
        super(VIPlannerNode, self).__init__()
        # config
        self.main_freq          = args.main_freq
        self.image_flip         = args.image_flip
        self.frame_id           = args.robot_id
        self.max_depth          = args.max_depth
        self.depth_uint_type    = args.depth_uint_type
        # ROS topics
        self.sem_topic          = args.sem_topic
        self.rgb_topic          = args.rgb_topic  
        self.depth_topic        = args.depth_topic
        self.depth_info_topic   = args.depth_cam_info_topic
        self.rgb_info_topic     = args.rgb_cam_info_topic
        self.timer_topic        = args.timer_topic
        self.compressed         = args.compressed
        # flags
        self.img_init: bool = False
        self.depth_intrinsics_init: bool = False
        self.rgb_intrinsics_init: bool = False

        # init buffers
        self.rgb_img: np.ndarray = None
        self.depth_img: np.ndarray = None
        self.K_depth: np.ndarray = np.zeros((3,3))
        self.K_rgb: np.ndarray = np.zeros((3,3))
        self.pix_depth_cam_frame: np.ndarray = None

        # init transforms
        self.tf_listener = tf.TransformListener()

        # init bridge
        self.bridge = cv_bridge.CvBridge()
        
        # init semantic network
        self.m2f_inference = Mask2FormerInference(
            config_file=args.m2f_config_path,
            model_weights=args.m2f_model_path,
        )
        
        # depth and rgb image message --> time syncronization by message_filters
        img_depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        if self.compressed:
            img_rgb_sub = message_filters.Subscriber(self.rgb_topic, CompressedImage)
        else:
            img_rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        ts = message_filters.TimeSynchronizer([img_depth_sub, img_rgb_sub], 20)
        if self.compressed:
            ts.registerCallback(self.imageCallbackCompressed)
        else:
            ts.registerCallback(self.imageCallback)
        
        # camera info subscribers
        rospy.Subscriber(self.depth_info_topic, CameraInfo, callback=self.depthCamInfoCallback)
        rospy.Subscriber(self.rgb_info_topic, CameraInfo, callback=self.rgbCamInfoCallback)
        
        # planning status topics
        self.sem_pub = rospy.Publisher(self.sem_topic, Image, queue_size=10)

        # semantic time
        self.timer_data = Float32()
        self.timer_pub = rospy.Publisher(self.timer_topic, Float32, queue_size=10)

        rospy.loginfo("Mask2Former Ready.")

    def spin(self):
        r = rospy.Rate(self.main_freq)
        while not rospy.is_shutdown():
            if self.img_init and self.depth_intrinsics_init and self.rgb_intrinsics_init:
                # main planning starts
                cur_rgb_image = self.rgb_img.copy()
                cur_depth_image = self.depth_img.copy()
                cur_depth_pose = self.depth_pose.copy()
                cur_rgb_pose = self.rgb_pose.copy()

                # crop rgb image
                start = time.time()
                if self.pix_depth_cam_frame is None:
                    self.initPixArray(cur_depth_image.shape)
                crop_rgb_image = self.imageWarp(cur_rgb_image, cur_depth_image, cur_rgb_pose, cur_depth_pose)
                # Estimate Semantics of RGB Image
                cur_sem_image = self.m2f_inference.predict(crop_rgb_image)
                # publish needed time
                time_need = start - time.time()
                self.timer_data.data = (time_need) * 1000
                self.timer_pub.publish(self.timer_data)
                # publish sem image
                sem_msg = self.bridge.cv2_to_imgmsg(cur_sem_image)
                sem_msg.header.stamp = self.image_time
                self.sem_pub.publish(sem_msg)
            r.sleep()
        rospy.spin()

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
        points = dep_im_reshaped * (depth_rot.T @ self.pix_depth_cam_frame.T).T + pose_depth[:3]
        
        # transform points to semantic camera frame
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
            plt.imshow(rgb_img)
            plt.figure(2)
            print(depth_img)
            plt.imshow(depth_img)
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(depth_img)
            ax2.imshow(rgb_warped / 255)
            ax3.imshow(depth_img)
            ax3.imshow(rgb_warped / 255, alpha=0.5)
            plt.show()    
        
        # reshape to image
        return rgb_warped

    def imageCallback(self, depth_msg: Image, rgb_msg: Image):
        rospy.logdebug("Received rgb   image %s: %d"%(rgb_msg.header.frame_id,   rgb_msg.header.seq))
        rospy.logdebug("Received depth image %s: %d"%(depth_msg.header.frame_id, depth_msg.header.seq))        

        # image time
        self.image_time = depth_msg.header.stamp
        
        # RGB image
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            if not self.image_flip:
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
            if not self.image_flip:
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
        if self.depth_uint_type:
            frame = frame / 1000.0
        frame[frame > self.max_depth] = 0.0
        # DEBUG - Visual Image
        # img = PIL.Image.fromarray((frame * 255 / np.max(frame[frame>0])).astype('uint8'))
        # img.show()
        if self.image_flip:
            frame = PIL.Image.fromarray(frame)
            self.depth_img = np.array(frame.transpose(PIL.Image.Transpose.ROTATE_180))
        else:
            self.depth_img = frame
        return
    
    def poseCallback(self, rgb_msg: Union[Image, CompressedImage], depth_msg: Image):
        # get current pose of semantic and depth image
        try:            
            pose = self.tf_listener.lookupTransform(self.frame_id, rgb_msg.header.frame_id, rgb_msg.header.stamp)
            self.rgb_pose = np.hstack(pose)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Fail to transfer the goal into base frame.")
        try:
            pose = self.tf_listener.lookupTransform(self.frame_id, depth_msg.header.frame_id, depth_msg.header.stamp)
            self.depth_pose = np.hstack(pose)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Fail to transfer the goal into base frame.")

        # declare ready for processing
        self.img_init = True
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

    node_name = "m2f_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument(
        'main_freq',       
        type=int,    
        default=5,                          
        help="frequency of path planner"
    )
    parser.add_argument(
        'image_flip',      
        type=bool,   
        default=False,                       
        help='is the image fliped'
    )
    parser.add_argument(
        'robot_id',        
        type=str,    
        default='base',                     
        help='robot TF frame id'
    )
    parser.add_argument(
        'max_depth',        
        type=int,    
        default=15,                     
        help='max depth for depth image'
    )
    parser.add_argument(
        'depth_uint_type',       
        type=bool,   
        default=False,                      
        help="image in uint type or not"
    )

    # ROS topics
    parser.add_argument(
        'sem_topic',      
        type=str,    
        default='/m2f_sem',                    
        help='Semantic Estimation topic'
    )
    parser.add_argument(
        'rgb_topic',       
        type=str,    
        default='/wide_angle_camera_front/image_raw/compressed',
        help='rgb camera topic'
    )
    parser.add_argument(
        'depth_topic',     
        type=str,    
        default='/depth_camera_front_upper/depth/image_rect_raw', 
        help='depth image ros topic'
    )
    parser.add_argument(
        'rgb_cam_info_topic',       
        type=str,    
        default='/wide_angle_camera_front/camera_info',
        help='rgb camera info topic (get intrinsic matrix)'
    )
    parser.add_argument(
        'depth_cam_info_topic',     
        type=str,    
        default='/depth_camera_front_upper/depth/camera_info', 
        help='depth image info topic (get intrinsic matrix)'
    )
    parser.add_argument(
        'timer_topic',     
        type=str,    
        default='/m2f_timer', 
        help='Time needed for semantic segmentation'
    )
    parser.add_argument(
        'compressed',     
        type=bool,    
        default=True, 
        help='If compressed rgb topic is used'
    )
    
    # Mask2FormerInferenceConfig
    parser.add_argument(
        'm2f_config_file', 
        type=str,    
        default='models/coco_panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml',   
        help="config file for m2f model"
    )
    parser.add_argument(
        'm2f_model_path',  
        type=str,    
        default='models/coco_panoptic/swin/model_final_9fd0ae.pkl',   
        help="read model"
    )
    
    args = parser.parse_args()
    args.m2f_config_path = os.path.join(pack_path, args.m2f_config_file)
    args.m2f_model_path  = os.path.join(pack_path, args.m2f_model_path)

    node = VIPlannerNode(args)

    node.spin()
