#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      Add semantic estimation of RGB Cameras to ROSBAG (further used for semantic point-cloud generation)
"""
# python
import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import tqdm

# ROS
import rosbag
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from rosbag_merge.bag_stream import main as bag_merger

# viplanner
from config import Mask2FormerCfg
from utils.m2f_utils import M2FWrapper


def main(args: argparse.Namespace, m2f_cfg: Mask2FormerCfg):
    # init wrapper
    m2f_wrapper = M2FWrapper(m2f_cfg)
    print(f"[INFO] Loaded Mask2Former model from {m2f_cfg.model_file} with config {m2f_cfg.config_file}.")
    # init cv_bridge
    bridge = CvBridge()
    
    if args.debug:
        # init plot to show predictions
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((1080, 1440, 3), dtype=np.uint8))
    
    # Open the input rosbag file
    input_bag = rosbag.Bag(args.bag_file, 'r')
    rgb_msg_count = sum([input_bag.get_message_count(rgb_msg) for rgb_msg in args.topic_rgb])
    pbar = tqdm.tqdm(total=(rgb_msg_count))
    print(f"[INFO] Opened rosbag {args.bag_file} with total of {input_bag.get_message_count()} messages and {rgb_msg_count} RGB messages.")
        
    # Create a new rosbag file for writing
    output_rosbag_file = args.bag_file[:-4] + '_modified.bag'
    with rosbag.Bag(output_rosbag_file, 'w') as output_bag:
        
        # Iterate through each message in the input rosbag file
        for topic, msg, t in input_bag.read_messages(topics=args.topic_rgb):
            
            # get BGR image
            if msg._type == Image._type:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            elif msg._type == CompressedImage._type:
                img = bridge.compressed_imgmsg_to_cv2(msg)
            else:
                raise ValueError("Unknown image type: {}".format(msg._type))
                            
            # Perform semantic estimation and get image with viplanner semantic labels in RGB format
            sem_img = m2f_wrapper.run_image(img=img)
            sem_img = cv2.cvtColor(sem_img, cv2.COLOR_RGB2BGR)
            
            if msg._type == Image._type:
                # Convert the modified image back to an image message
                modified_msg = Image()
                modified_msg.header = msg.header
                modified_msg.encoding = "bgr8"
                modified_msg.height = sem_img.shape[0]
                modified_msg.width = sem_img.shape[1]
                # modified_msg.step = sem_img.shape[1]
                modified_msg.data = np.array(sem_img).tobytes()
            else:
                # Convert the modified image back to a compressed image message
                modified_msg = CompressedImage()
                modified_msg.header = msg.header
                modified_msg.format = msg.format
                modified_msg.data = np.array(cv2.imencode('.jpg', sem_img)[1]).tobytes()
            
            # Write the modified image message to the output rosbag file under a new topic
            output_bag.write(topic + "_sem", modified_msg, t)

            if args.debug:
                # show prediction
                im.set_data(sem_img[:, :, ::-1])
                plt.pause(0.001)
                plt.draw()
                
            # update progress bar
            pbar.update(1)
            pbar.set_description(f'Processing messages')
    
    # close bag and progress bar
    input_bag.close()
    pbar.close()
    
    # merge bags
    bag_folder, _ = os.path.split(args.bag_file)
    merged_bag = args.bag_file[:-4] + '_merged.bag'
    bag_merger(
        input_bags=[args.bag_file, output_rosbag_file],
        topics=[],  # all topics
        output_path=bag_folder,
        outbag_name=merged_bag,
        write_csvs=False,
        write_bag=True,
    )
                    
    # Rename the output rosbag file to replace the input file
    os.remove(args.bag_file)
    os.remove(output_rosbag_file)
    os.rename(merged_bag, args.bag_file)
    
    print("[INFO] Rosbag modification complete.")
    return


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Predict Semantic Labels for RGB Images in Rosbag")
    parser.add_argument("-bf", "--bag_file", help="Input ROS bag.",
                        default='/home/pascal/SemNav/env/anymal/2023_03_27_eth_sun/mergedBag.bag')
    parser.add_argument("-c", "--topic_rgb", nargs="+", help="Image topics.",
                        default=[
                            "/alphasense_driver_ros/cam3/color/image/compressed",  # left
                            "/alphasense_driver_ros/cam4/color/image/compressed",  # center
                            "/alphasense_driver_ros/cam5/color/image/compressed",  # right
                        ])
    parser.add_argument("-d", "--debug", action="store_true", help="Visualize semantic prediction and debug information (assumes img of size 1080x1440)",
                        default=False)
    
    args = parser.parse_args()    
    
    # Mask2Former args
    m2f_cfg = Mask2FormerCfg()
    
    # main
    main(args, m2f_cfg)

# EoF
