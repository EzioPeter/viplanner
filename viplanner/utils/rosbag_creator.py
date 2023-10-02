import os

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
from rospy import rostime
from std_msgs.msg import Header

# Create a CvBridge to convert images to ROS messages
bridge = CvBridge()


def write_to_bag(bag, topic, dir_name, bag_dir, time_stamps, encoding="bgr8"):
    img_list = sorted(os.listdir(os.path.join(bag_dir, dir_name)))

    for idx, img in enumerate(img_list):
        # Construct the image file path
        image_path = os.path.join(bag_dir, dir_name, img)

        # Read the image
        image = cv2.imread(image_path)

        if image is not None:
            # Convert the image to a ROS Image message
            image_msg = bridge.cv2_to_imgmsg(image, encoding=encoding)

            # Create a ROS Header with the timestamp
            header = Header()
            img_idx = int(img[6:11])
            timestamp = float(time_stamps[img_idx, 0] + time_stamps[img_idx, 1] * 10e-9)
            header.stamp = rostime.Time.from_sec(timestamp)
            # Set the frame ID
            header.frame_id = "odom"
            # Set the image header
            image_msg.header = header

            # Write the image message to the bag
            bag.write(topic, image_msg, header.stamp)


def generate_rosbag(bag_dir, output_bag_file):
    # get image timestamps
    time_rgb = np.loadtxt(os.path.join(bag_dir, "odom_bgr.txt"))[:, 7:]
    time_depth = np.loadtxt(os.path.join(bag_dir, "odom_depth.txt"))[:, 7:]
    time_sem = np.loadtxt(os.path.join(bag_dir, "odom_sem.txt"))[:, 7:]

    if os.path.exists(os.path.join(bag_dir, output_bag_file)):
        os.remove(os.path.join(bag_dir, output_bag_file))

    # Initialize the ROS bag
    bag = rosbag.Bag(os.path.join(bag_dir, output_bag_file), "w")

    # Write the images to the bag
    write_to_bag(bag, "/path_projected/viplanner/rgb", "video_rgb_projected", bag_dir, time_rgb)
    write_to_bag(bag, "/path_projected/viplanner/depth", "video_depth_projected", bag_dir, time_depth)
    write_to_bag(bag, "/path_projected/viplanner/sem", "video_sem_projected", bag_dir, time_sem)
    write_to_bag(bag, "/path_projected/viplanner/sem_all", "video_sem_after_projected", bag_dir, time_rgb)
    write_to_bag(bag, "/path_projected/iplanner/depth", "video_depth_iplanner_projected", bag_dir, time_depth)

    # Close the ROS bag
    bag.close()


if __name__ == "__main__":
    bag_dir = "/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_crosswalk_sidewalk_wet_success"
    # bag_dir = "/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_stairs_both_door"
    output_bag_file = "path_projected.bag"

    generate_rosbag(bag_dir, output_bag_file)
