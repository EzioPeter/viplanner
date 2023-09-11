"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      Add semantic estimation of RGB Cameras to ROSBAG (further used for semantic point-cloud generation)
"""
import argparse

# python
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ROS
import rosbag
import tqdm
from cv_bridge import CvBridge
from rosbag_merge.bag_stream import main as bag_merger
from sensor_msgs.msg import CompressedImage, Image

# viplanner
from viplanner.config import Mask2FormerCfg
from viplanner.intern.m2f.m2f_utils import M2FWrapper


def main(args: argparse.Namespace, m2f_cfg: Mask2FormerCfg):
    # init wrapper
    m2f_wrapper = M2FWrapper(m2f_cfg)
    print(
        f"[INFO] Loaded Mask2Former model from {m2f_cfg.model_file} with"
        f" config {m2f_cfg.config_file}."
    )
    # init cv_bridge
    bridge = CvBridge()

    if args.debug:
        # init plot to show predictions
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((1080, 1440, 3), dtype=np.uint8))

    # Open the input rosbag file
    input_bag = rosbag.Bag(args.bag_file, "r")
    rgb_msg_count = sum(
        [input_bag.get_message_count(rgb_msg) for rgb_msg in args.topic_rgb]
    )
    pbar = tqdm.tqdm(total=(rgb_msg_count))
    print(
        f"[INFO] Opened rosbag {args.bag_file} with total of"
        f" {input_bag.get_message_count()} messages and {rgb_msg_count} RGB"
        " messages."
    )

    # Create a new rosbag file for writing
    output_rosbag_file = args.bag_file[:-4] + "_modified.bag"
    if os.path.exists(output_rosbag_file):
        os.remove(output_rosbag_file)
    output_bag = rosbag.Bag(output_rosbag_file, "w")
    img_counter = 0

    # Iterate through each message in the input rosbag file
    for topic, msg, t in input_bag.read_messages(topics=args.topic_rgb):
        # get BGR image
        if msg._type == Image._type:
            img = cv2.imdecode(
                np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR
            )
        elif msg._type == CompressedImage._type:
            img = bridge.compressed_imgmsg_to_cv2(msg)
            if "bayer_rggb8" in msg.format:
                img = cv2.cvtColor(img, cv2.COLOR_BayerRGGB2BGR)
        else:
            raise ValueError(f"Unknown image type: {msg._type}")

        # Perform semantic estimation and get image with viplanner semantic labels in RGB format
        sem_img = m2f_wrapper.run_image(img=img)
        sem_img = cv2.cvtColor(sem_img, cv2.COLOR_RGB2BGR)
        # sem_img = cv2.imread("/home/pascal/viplanner/env/anymal/2023_03_23_rsl/semantics/frame_" + str(img_counter).zfill(5) + ".png")

        # Convert the modified image back to an image message
        if args.compressed:
            # Convert the image to JPEG format
            success, compressed_image = cv2.imencode(".jpg", sem_img)
            if not success:
                raise RuntimeError("Failed to encode image.")

            modified_msg = CompressedImage()
            modified_msg.header = msg.header
            modified_msg.format = "jpeg"
            modified_msg.data = np.array(compressed_image).tostring()
        else:
            modified_msg = Image()
            modified_msg.header = msg.header
            modified_msg.encoding = "bgr8"
            modified_msg.height = sem_img.shape[0]
            modified_msg.width = sem_img.shape[1]
            modified_msg.step = sem_img.shape[1] * 3
            modified_msg.data = np.array(sem_img).tobytes()

        # Write the modified image message to the output rosbag file under a new topic
        if topic.endswith("/compressed"):
            topic = topic[: -len("/compressed")] + "_sem" + "/compressed"
        else:
            topic = topic + "_sem"
        output_bag.write(topic, modified_msg, t)

        if args.debug:
            # show prediction
            im.set_data(sem_img[:, :, ::-1])
            plt.pause(0.001)
            plt.draw()

        # update progress bar
        pbar.update(1)
        pbar.set_description(f"Processing messages")

        img_counter += 1

    # close bag and progress bar
    output_bag.close()
    input_bag.close()
    pbar.close()

    # merge bags
    bag_folder, _ = os.path.split(args.bag_file)
    merged_bag = args.bag_file[:-4] + "_merged"
    bag_merger(
        input_bags=[args.bag_file, output_rosbag_file],
        topics=[],  # all topics
        output_path=bag_folder,
        outbag_name=merged_bag,
        write_csvs=False,
        write_bag=True,
    )

    # cleanup
    os.remove(output_rosbag_file)

    print("[INFO] Rosbag modification complete.")
    return


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(
        description="Predict Semantic Labels for RGB Images in Rosbag"
    )
    parser.add_argument(
        "-bf",
        "--bag_file",
        help="Input ROS bag.",
        default="/home/pascal/viplanner/env/anymal/2023_01_26_eth/mission_0_all.bag",
    )  # '/home/pascal/viplanner/env/anymal/2023_03_27_eth_sun/mergedBag.bag')
    parser.add_argument(
        "-t",
        "--topic_rgb",
        nargs="+",
        help="Image topics.",
        default=[
            # "/alphasense_driver_ros/cam3/color/image/compressed",  # left
            # "/alphasense_driver_ros/cam4/color/image/compressed",  # center
            # "/alphasense_driver_ros/cam5/color/image/compressed",  # right
            # "/wide_angle_camera_front/image_raw/compressed",
            "/wide_angle_camera_front/image_color_rect/compressed",
        ],
    )
    parser.add_argument(
        "-c",
        "--compressed",
        action="store_false",
        help="Store the image in the new bag as compressed image",
        default=True,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_false",
        help=(
            "Visualize semantic prediction and debug information (assumes img"
            " of size 1080x1440)"
        ),
        default=True,
    )

    args = parser.parse_args()

    # Mask2Former args
    m2f_cfg = Mask2FormerCfg()

    # main
    main(args, m2f_cfg)

# EoF
