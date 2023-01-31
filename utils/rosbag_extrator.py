#!/usr/bin/env python3
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Extract Images from ROSbags
"""

# python
import random
import os
import cv2
import numpy as np
import argparse

# ROS
import rosbag

def main(args):
    """
    Extract a folder of images from a rosbag.
    """
    print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # if total number of images defined, select them randomely from rosbag
    message_count = bag.get_message_count(args.image_topic)   
    if args.nb_images:
        assert message_count > args.nb_images, 'Should select %s but only total of %s included in bag!' % (args.nb_images, message_count)
        image_idx = random.sample(range(message_count), args.nb_images)
    else:
        image_idx = range(message_count)

    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):

        if count in image_idx:
            if args.compressed:
                im = np.fromstring(msg.data, np.uint8)
                cv_img = cv2.imdecode(im, cv2.IMREAD_COLOR)
            else:        
                im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                cv_img = cv2.cvtColor(im, cv2.IMREAD_COLOR)
                # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr16")

            print("Extracted image %i" % count)
            cv2.imwrite(os.path.join(args.output_dir, "frame%06i.jpg" % count), cv_img)
            
        count += 1

    bag.close()

    return


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-bf", "--bag_file", default='/home/pascal/SemNav/sem_seg/data/rosbags/2023-01-13-16-01-31_anymal-d020-npc_mission_0.bag', # required=True, 
                        help="Input ROS bag.")
    parser.add_argument("-o", "--output_dir", default='/home/pascal/SemNav/sem_seg/data/rosbags/2023-01-13-16-01-31_anymal-d020-npc_mission_0', # required=True, 
                        help="Output directory.")
    parser.add_argument("-t", "--image_topic", default='/wide_angle_camera_front/image_color_rect/compressed', # required=True, 
                        help="Image topic.")
    parser.add_argument("-n", "--nb_images", type=int, default=1000, 
                        help="Total number of image extracted from the ROS bag")
    parser.add_argument('-c', "--compressed", action='store_false', 
                        help='Compressed images within rosbag')
    args = parser.parse_args()

    main(args)
