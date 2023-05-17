#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      Extract Real World Data from Rosbags in the format expected by the eval_real.py script
"""

# import packages
import os
import cv2
import numpy as np
import argparse
from typing import List
import tqdm

# ROS
import rospy
import rosbag
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import tf2_ros
import tf2_py as tf2
from rosgraph import rosenv
import roslaunch
import time

def check_roscore():
    # Check if roscore is already running
    try:
        master_uri = rosenv.get_master_uri()
    except rospy.service.ServiceException:
        master_uri = None

    # If roscore is not running, start a new instance
    if master_uri is None:
        print("roscore is not running, starting a new instance...")
        roscore_launch = roslaunch.scriptapi.ROSLaunch()
        roscore_launch.start()
        roscore_process = roscore_launch.launch(roslaunch.pmon.REQUIRED, roslaunch.rlutil.resolve_launch_arguments(["roscore"]))
        master_uri = rosenv.get_master_uri()

        # Wait for roscore to start up
        while master_uri is None:
            time.sleep(1)
            try:
                master_uri = rosenv.get_master_uri()
            except rospy.service.ServiceException:
                pass

        print("roscore started successfully.")
    else:
        print("roscore is already running.")

    return master_uri

def setup_tf_buffer(bag: rosbag.Bag):
    """
    Gets the transform from frame A to frame B from a rosbag /tf message at a specific timestamp.

    Parameters:
        bag (rosbag.Bag): Rosbag with a /tf message.

    Returns:
        tf_buffer (tf2_ros.Buffer): A tf buffer with the transforms from the rosbag.
    """
    print("Setting up tf buffer...", end=" ")
    tf_buffer = tf2_ros.Buffer(rospy.Duration(bag.get_end_time() - bag.get_start_time()))
    for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
        for msg_tf in msg.transforms:
            if topic == '/tf_static':
                tf_buffer.set_transform_static(msg_tf, "default_authority")
            else:
                tf_buffer.set_transform(msg_tf, "default_authority")
    print("Done")
    return tf_buffer
    
def get_intrinsics(bag: rosbag.Bag, topics: List[str], image_topic: str) -> np.ndarray:
    """
    Get camera intrinsics from rosbag
    """
    topic_type = topics[image_topic].msg_type
    
    if topic_type == Image._type:
        cam_parent = os.path.split(image_topic)[0]
        camera_info_topic = f"{cam_parent}/camera_info"
        # Check if the CameraInfo topic is in the list of topics
        if camera_info_topic in topics:
            # Print a message indicating that the CameraInfo topic was found
            print(f"Corresponding CameraInfo topic found: {camera_info_topic}")
            for _, msg, _ in bag.read_messages(topics=[camera_info_topic]):
                K = np.array(msg.K).reshape(3, 3)
                height = msg.height
                width  = msg.width
                break
            assert K is not None, f'Could not find camera info under topic {camera_info_topic}'
        else:
            # Print a message indicating that the CameraInfo topic was not found
            print(f"Corresponding CameraInfo topic not found for {image_topic}")
    elif topic_type == CompressedImage._type:
        cam_parent = image_topic.split('/')[1]
        camera_info_topic = f"/{cam_parent}/camera_info"
        # Check if the CameraInfo topic is in the list of topics
        if camera_info_topic in topics:
            # Print a message indicating that the CameraInfo topic was found
            print(f"Corresponding CameraInfo topic found: {camera_info_topic}")
            for _, msg, _ in bag.read_messages(topics=[camera_info_topic]):
                K = np.array(msg.K).reshape(3, 3)
                height = msg.height
                width  = msg.width
                break
            assert K is not None, f'Could not find camera info under topic {camera_info_topic}'
        else:
            # Print a message indicating that the CameraInfo topic was not found
            print(f"Corresponding CameraInfo topic not found for {image_topic}")
    else:
        raise ValueError(f'Topic {image_topic} is of type {topic_type} which is not supported!')
    
    return K, height, width


def main(args):
    """
    Extract a folder of images from a rosbag.
    """
    bag = rosbag.Bag(args.bag_file, "r")
    print(f"Opened rosbag {args.bag_file} with {bag.get_message_count()} messages")
    topics = bag.get_type_and_topic_info().topics
    
    # init cv_bridge
    bridge = CvBridge()
    
    # check if topic in rosbag
    if any([used_topic not in topics for used_topic in [args.topic_depth, args.topic_bgr, args.topic_state]]):
        raise ValueError('Not all topics in bag!')
    
    # init buffers
    odom_base = np.ndarray((bag.get_message_count(args.topic_state), 9))  # x, y, z, qx, qy, qz, qw, t_sec, t_nsec
    odom_depth = np.ndarray((bag.get_message_count(args.topic_depth), 9))  # x, y, z, qx, qy, qz, qw, t_sec, t_nsec
    odom_bgr = np.ndarray((bag.get_message_count(args.topic_bgr), 9))  # x, y, z, qx, qy, qz, qw, t_sec, t_nsec
    odom_goal = np.ndarray((bag.get_message_count(args.topic_goal), 5))  # x, y, z, t_sec, t_nsec
    
    # get intrinsics
    K_depth, height_depth, width_depth = get_intrinsics(bag, topics, args.topic_depth)
    K_bgr, _, _ = get_intrinsics(bag, topics, args.topic_bgr)
    
    # make directory structure for output
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/bgr", exist_ok=True)
    os.makedirs(args.output_dir + "/depth", exist_ok=True)
    
    # save intrinsics
    np.savetxt(os.path.join(args.output_dir, "intrinsics_depth.txt"), K_depth)
    np.savetxt(os.path.join(args.output_dir, "intrinsics_bgr.txt"), K_bgr)
    
    # init counters
    bgr_counter = 0
    depth_counter = 0
    state_counter = 0
    goal_counter = 0
    
    # get transform between odom and cameras
    check_roscore()  # roscore needs to run to use tf_buffer
    tf_buffer = setup_tf_buffer(bag)
    
    # configure process bar
    pbar = tqdm.tqdm(total=(bag.get_message_count(args.topic_depth) + bag.get_message_count(args.topic_bgr) + bag.get_message_count(args.topic_state) + bag.get_message_count(args.topic_goal)))
    
    for topic, msg, t in bag.read_messages(topics=[args.topic_depth, args.topic_bgr, args.topic_state, args.topic_goal]):
        topic_type = msg._type
        if topic_type == Image._type:  # DEPTH
            try:
                transform_stamp = tf_buffer.lookup_transform("odom", msg.header.frame_id, t)
            except tf2.ExtrapolationException:
                continue
            
            im = np.frombuffer(msg.data, dtype=np.uint16).reshape(height_depth, width_depth, -1)
            im = cv2.rotate(im, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(args.output_dir, "depth", "frame_" + f"{int(depth_counter)}".zfill(5) + ".png"), im)
            odom_depth[depth_counter, :] = np.array([
                transform_stamp.transform.translation.x,
                transform_stamp.transform.translation.y,
                transform_stamp.transform.translation.z,
                transform_stamp.transform.rotation.x,
                transform_stamp.transform.rotation.y,
                transform_stamp.transform.rotation.z,
                transform_stamp.transform.rotation.w,
                t.secs,
                t.nsecs
            ])
            depth_counter += 1
            
        elif topic_type == CompressedImage._type: # BGR
            try:
                transform_stamp = tf_buffer.lookup_transform("odom", msg.header.frame_id, t)
            except tf2.ExtrapolationException:
                continue
            
            im = bridge.compressed_imgmsg_to_cv2(msg)
            if "bayer_rggb8" in msg.format:
                im = cv2.cvtColor(im, cv2.COLOR_BayerRGGB2BGR)
            cv2.imwrite(os.path.join(args.output_dir, "bgr", "frame_" + f"{int(bgr_counter)}".zfill(5) + ".png"), im)
            odom_bgr[bgr_counter, :] = np.array([
                transform_stamp.transform.translation.x,
                transform_stamp.transform.translation.y,
                transform_stamp.transform.translation.z,
                transform_stamp.transform.rotation.x,
                transform_stamp.transform.rotation.y,
                transform_stamp.transform.rotation.z,
                transform_stamp.transform.rotation.w,
                t.secs,
                t.nsecs
            ])
            bgr_counter += 1
        
        elif topic_type == Odometry._type:  # STATE
            odom_base[state_counter] = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
                t.secs,
                t.nsecs
            ])
            state_counter += 1
        
        elif topic_type == PointStamped._type:  # GOAL / WAYPOINT        
            odom_goal[goal_counter, :] = np.array([
                msg.point.x,
                msg.point.y,
                msg.point.z,
                t.secs,
                t.nsecs
            ])
            goal_counter += 1

        # update progress bar
        pbar.update(1)
        pbar.set_description(f'Processing messages (bgr: {bgr_counter}, depth: {depth_counter}, state: {state_counter}, goal: {goal_counter})')
               
    bag.close()
    pbar.close()
    
    # save timestamps
    np.savetxt(os.path.join(args.output_dir, "odom_depth.txt"), odom_depth[:depth_counter])
    np.savetxt(os.path.join(args.output_dir, "odom_bgr.txt"), odom_bgr[:bgr_counter])
    np.savetxt(os.path.join(args.output_dir, "odom_base.txt"), odom_base[:state_counter])
    np.savetxt(os.path.join(args.output_dir, "odom_goal.txt"), odom_goal[:goal_counter])
    print("SUCCESS")
    return


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="Extract msgs from a ROS bag.")
    parser.add_argument("-bf", "--bag_file", default='/home/pascal/SemNav/env/anymal/2023_03_23_rsl/_2023-03-23-20-30-17.bag',
                        help="Input ROS bag.")
    parser.add_argument("-o", "--output_dir", default='/home/pascal/SemNav/env/anymal/2023_03_23_rsl/',
                        help="Output directory.")
    parser.add_argument("-td", "--topic_depth", default='/depth_camera_front_upper/depth/image_rect_raw',
                        help="Image topic.")
    parser.add_argument("-tr", "--topic_bgr", default='/wide_angle_camera_front/image_raw/compressed',
                        help="Image topic.")
    parser.add_argument("-ts", "--topic_state", default="/state_estimator/odometry",
                        help="Image topic.")
    parser.add_argument("-tg", "--topic_goal", default="/mp_waypoint",
                        help="Topic where the way-/goalpoints are published.")
    args = parser.parse_args()

    main(args)

# EoF
