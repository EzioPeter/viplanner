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

# ROS
import rospy
import rospkg
import tf
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import ros_numpy

# init ros node
rospack = rospkg.RosPack()
pack_path = rospack.get_path('viplanner_node')
sys.path.append(pack_path)

# visual imperative planner
from model_src.m2f_inference import Mask2FormerInference
from utils.rosutil import ROSArgparse


class VIPlannerNode:
    """VIPlanner ROS Node Class"""
    def __init__(self, args):
        super(VIPlannerNode, self).__init__()
        # config
        self.main_freq          = args.main_freq
        self.image_flip         = args.image_flip
        self.frame_id           = args.robot_id
        # ROS topics
        self.sem_topic          = args.sem_topic
        self.rgb_topic          = args.rgb_topic  
        self.depth_topic        = args.depth_topic
        self.depth_info_topic   = args.depth_cam_info_topic
        self.sem_info_topic     = args.sem_cam_info_topic
        self.timer_topic        = args.timer_topic
        # flags
        self.img_init: bool = False
        self.depth_intrinsics_init: bool = False
        self.sem_intrinsics_init: bool = False

        # init buffers
        self.rgb_img: np.ndarray = None
        self.depth_img: np.ndarray = None
        self.K_depth: np.ndarray = np.zeros((3,3))
        self.K_sem: np.ndarray = np.zeros((3,3))
        self.pix_depth_cam_frame: np.ndarray = None

        # init transforms
        self.tf_listener = tf.TransformListener()

        # init semantic network
        self.m2f_inference = Mask2FormerInference(
            config_file=args.m2f_config_path,
            model_weights=args.m2f_model_path,
        )
        
        # depth and rgb image message --> time syncronization by message_filters
        img_depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        img_rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        ts = message_filters.TimeSynchronizer([img_depth_sub, img_rgb_sub], 10)
        ts.registerCallback(self.imageCallback)
        
        # camera info subscribers
        rospy.Subscriber(self.depth_info_topic, CameraInfo, callback=self.depthCamInfoCallback)
        rospy.Subscriber(self.sem_intrinsics_init, CameraInfo, callback=self.semCamInfoCallback)
        
        # planning status topics
        self.sem_pub = rospy.Publisher(self.sem_topic, Image, queue_size=10)

        # semantic time
        self.timer_data = Float32()
        self.timer_pub = rospy.Publisher(self.timer_topic, Float32, queue_size=10)

        rospy.loginfo("Mask2Former Ready.")

    def spin(self):
        r = rospy.Rate(self.main_freq)
        while not rospy.is_shutdown():
            if self.img_init and self.depth_intrinsics_init and self.sem_intrinsics_init:
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
                img_sem = Image()
                img_sem.data = cur_sem_image
                img_sem.header.stamp = self.image_time
                self.sem_pub(img_sem)
            r.sleep()
        rospy.spin()

    def initPixArray(self, img_shape: tuple):
        # init pixel array
        self.pix_depth_cam_frame = np.zeros((img_shape[0], img_shape[1], 3))
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                self.pix_depth_cam_frame[i, j, :] = np.array([i, j, 1])
        self.pix_depth_cam_frame = self.pix_depth_cam_frame.reshape(-1, 3).T
        return
        
    def imageWarp(self, rgb_img: np.ndarray, depth_img: np.ndarray, pose_rgb: np.ndarray, pose_depth: np.ndarray) -> np.ndarray:
        # get 3D points of depth image
        depth_rot = tf.Rotation.from_quat(pose_depth[3:]).as_matrix()
        dep_im_reshaped = np.flipud(depth_img.reshape(-1, 1))  # flip s.t. start in lower left corner of image as (0,0) -> has to fit to the pixel tensor
        points = dep_im_reshaped * (depth_rot.T @ self.pix_depth_cam_frame.T).T + pose_depth[:3]
        
        # transform points to semantic camera frame
        points_sem_cam_frame = (tf.Rotation.from_quat(pose_rgb[3:]).as_matrix() @ (points - pose_rgb[:3]).T).T
        # normalize points
        points_sem_cam_frame_norm = points_sem_cam_frame / points_sem_cam_frame[:, 0][:, np.newaxis]
        # reorder points be camera convention (z-forward)
        points_sem_cam_frame_norm = points_sem_cam_frame_norm[:, [1, 2, 0]]  * np.array([-1, -1, 1])
        # transform points to pixel coordinates
        pixels = (self.K_sem @ points_sem_cam_frame_norm.T).T
        # filter points outside of image
        filter_idx = (pixels[:, 0] >= 0) & (pixels[:, 0] < rgb_img.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < rgb_img.shape[0])
        # get semantic annotation
        sem_annotation = np.zeros((pixels.shape[0], 3))
        sem_annotation[filter_idx] = rgb_img[pixels[filter_idx, 1].astype(int)-1, pixels[filter_idx, 0].astype(int)-1]
        sem_annotation = np.flipud(sem_annotation)
        # reshape to image
        return sem_annotation.reshape(depth_img.shape[0], depth_img.shape[1], 3)

    def imageCallback(self, depth_msg: Image, rgb_msg: Image):
        rospy.loginfo("Received rgb image %s: %d"%(rgb_msg.header.frame_id, rgb_msg.header.seq))
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
        frame = ros_numpy.numpify(rgb_msg)
        # DEBUG - Visual Image
        # img = PIL.Image.fromarray((frame)).astype('uint8'))
        # img.show()
        self.rgb_img = frame
        
        # get current pose of semantic and depth image
        try:
            rgb_msg.header.stamp = self.tf_listener.getLatestCommonTime(rgb_msg.header.frame_id, self.frame_id)
            # TODO: what transform to use? camera center?
            self.rgb_pose = self.tf_listener.transformPoint(self.frame_id, rgb)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Fail to transfer the goal into base frame.")
        try:
            depth_msg.header.stamp = self.tf_listener.getLatestCommonTime(depth_msg.header.frame_id, self.frame_id)
            # TODO: what transfrom to use? 
            self.depth_pose = self.tf_listener.transformPoint(self.frame_id, depth)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Fail to transfer the goal into base frame.")

        # declare ready for processing
        self.img_init = True
        return

    def depthCamInfoCallback(self, cam_info_msg: CameraInfo):
        if not self.depth_intrinsics_init:
            self.K_depth = cam_info_msg.K
            self.depth_intrinsics_init = True
        return

    def semCamInfoCallback(self, cam_info_msg: CameraInfo):
        if not self.sem_intrinsics_init:
            self.K_sem = cam_info_msg.K
            self.sem_intrinsics_init = True
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
        default=True,                       
        help='is the image fliped'
    )
    parser.add_argument(
        'robot_id',        
        type=str,    
        default='base',                     
        help='robot TF frame id'
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
        default='/rgbd_camera/depth/image', 
        help='depth image ros topic'
    )
    parser.add_argument(
        'rgb_cam_info_topic',       
        type=str,    
        default='/wide_angle_camera_front/image_raw/camerainfo',
        help='rgb camera info topic (get intrinsic matrix)'
    )
    parser.add_argument(
        'depth_cam_info_topic',     
        type=str,    
        default='/rgbd_camera/depth/camerainfo', 
        help='depth image info topic (get intrinsic matrix)'
    )
    parser.add_argument(
        'timer_topic',     
        type=str,    
        default='/m2f_timer', 
        help='Time needed for semantic segmentation'
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
