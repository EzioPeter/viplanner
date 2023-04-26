#!/usr/bin python3

"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      transform camera pose given as combination of robot odom and offset to camera extrinsic that contains both
"""

# python
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# parameters
odom_file_path = "/home/pascal/SemNav/env/matterport/data_domains/2n8kARJN3HM_fan/odom_ground_truth.txt"
extrinsic_path = "/home/pascal/SemNav/env/matterport/data_domains/2n8kARJN3HM_fan/camera_extrinsic_old.txt"
intrinsic_path = "/home/pascal/SemNav/env/matterport/data_domains/2n8kARJN3HM_fan/intrinsics.txt"
output_path = "/home/pascal/SemNav/env/matterport/data_domains/2n8kARJN3HM_fan/camera_extrinsic.txt"

"""Helper Functions"""

def read_extrinsics(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        elems = np.fromstring(lines[0][1:-1], dtype=float, sep=', ')
    CT = np.array(elems[:3])
    CR = R.from_quat(elems[3:])
    return CR, CT


def read_odom(odom_path):
    odom_list = []
    with open(odom_path) as f:
        lines = f.readlines()
        for line in lines:
            odom = np.fromstring(line[1:-1], dtype=float, sep=', ')
            odom_list.append(list(odom))
    return odom_list

def read_intrinsics(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        elems = np.fromstring(lines[0][1:-1], dtype=float, sep=', ')
    P = np.array(elems)
    return P
    
"""Main Function"""

def main():      
    odom_list = read_odom(odom_path=odom_file_path)
    cameraR, cameraT = read_extrinsics(extrinsic_path)

    # output array
    odom_extrinsics = np.zeros((len(odom_list), 7))
                               
    for idx, odom in enumerate(odom_list):
        # Rc = R.from_quat(odom[3:]) * cameraR
        C = (odom[:3] + cameraT)
        odom_extrinsics[idx, :3] = C
        odom_extrinsics[idx, 3:] = odom[3:]  # TODO Change back to Rc.as_quat()
    
    np.savetxt(output_path, odom_extrinsics, delimiter=',')
    
    # convert intrinsics to numpy save txt format
    P = read_intrinsics(intrinsic_path)
    np.savetxt(intrinsic_path, P, delimiter=',')
    
    # convert depth image names to new convention (add leading zeros)
    env_dir, _ = os.path.split(odom_file_path)
    depth_image_dir = os.path.join(env_dir, 'depth')
    depth_image_paths = os.listdir(depth_image_dir)

    for img_path in depth_image_paths:
        name, ext = os.path.splitext(img_path)
        
        if ext != '.png':
            continue
        
        os.rename(os.path.join(depth_image_dir, img_path), os.path.join(depth_image_dir, name.zfill(4) + ext))

if __name__ ==  '__main__':
    main()

# EoF
       