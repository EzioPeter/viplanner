#!/usr/bin python3

"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch
@author     Fan Yang
@email      fanyang1@ethz.ch

@brief      reconstruct 3D structure with depth images
"""

# python
import os
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import shutil

# imperative-cost-map
from config import ReconstructionCfg
from third_party.ip_basic.ip_basic.depth_map_utils import fill_in_multiscale


class DepthReconstruction:
    """
    Reconstruct 3D Map with depth images, assumes the ground truth camera odom is known
    Config parameters can be set in ReconstructionCfg

    Expects following datastructure:

    - env_name
        - camera_extrinsic.txt  (format: x y z qx qy qz qw)
        - intrinsics.txt (expects ROS CameraInfo format --> P-Matrix)
        - depth
            - xxxx.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
        - semantics (optional)
            - xxxx.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
    """
    def __init__(self, cfg: ReconstructionCfg):
        # get config
        self._cfg: ReconstructionCfg = cfg
        # read camera params and odom
        self.K: np.ndarray = None
        self._read_intrinsic()
        self.extrinsics_list: list = None
        self._read_extrinsic()
        # control flag if point-cloud has been loaded
        self._is_constructed =  False
        
        # variables
        self._points: np.ndarray = None
        self._pcd: o3d.geometry.PointCloud = None
        
        print("Ready to read depth data.")
        return None

    # public methods
    def depthMapReconstruction(self):
        # identify start and end image idx for the reconstruction
        N = len(self.extrinsics_list)
        if self._cfg.max_images:
            self._start_idx = self._cfg.start_idx
            self._end_idx = min(self._cfg.start_idx + self._cfg.max_images, N)
        else:
            self._start_idx = 0
            self._end_idx = N
        
        # load images
        self.depth_img = self._load_depth_images()
        if self._cfg.semantics:
            self.semantic_img = self._load_semantic_images()
            assert self.depth_img.shape[:3] == self.semantic_img.shape[:3], f"depth and semantic images should have the same shape, but got depth: {self.depth_img.shape} and semantic: {self.semantic_img.shape}"
        print(f"total number of images for reconstruction: {self.depth_img.shape[0]}")
        
        # init pixel tensor and point arrays
        x_nums, y_nums = self.depth_img.shape[1], self.depth_img.shape[2]
        T = self._computePixelTensor(x_nums, y_nums)
        pixel_nums = x_nums * y_nums
        self._points = []
        if self._cfg.semantics:
            self._sem_mapping = []
        
        # 3D reconstruction from Depth 
        k_inv = np.linalg.inv(self.K)     
        for img_idx in tqdm(range(self.depth_img.shape[0]), desc="Reconstructing 3D Points"):
            im = self.depth_img[img_idx]
            extrinsics = self.extrinsics_list[img_idx+self._start_idx]
            
            # apply rotation
            rot = R.from_quat(extrinsics[3:]).as_euler("XYZ", degrees=True)
            rot_transformed = np.zeros_like(rot)
            rot_transformed[1] = rot[2]  # rotation around the z axis in robotics frame  # NOTE: remove minus if FAN's data is used
            rot_transformed[0] = -rot[1]  # rotation around the y axis in robotics frame
            rot_mat = R.from_euler("XYZ", rot_transformed, degrees=True).as_matrix()
            T_z = T.reshape(3, -1)
            T_z = T_z[[1, 0, 2], :]  # reorder to be in camera frame (z forward, x right, y down)
            
            im_reshape = im.reshape(-1, 1)
            points = im_reshape * (rot_mat.T @ k_inv @ T_z).T
            points = points[:, [2, 0, 1]] * np.array([[1, -1, -1]])  # reorder to be in "robotics" frame (x forward, y left, z up)
            
            # filter points with 0 depth --> otherwise obstacles at camera position
            non_zero_idx = np.where(points.any(axis=1))[0]
            
            points_final = points[non_zero_idx] + extrinsics[:3]
            self._points.append(points_final)
            
            # create semantic mapping for the points
            if self._cfg.semantics:
                sem_im = self.semantic_img[img_idx].reshape(-1, 3)
                self._sem_mapping.append(sem_im[non_zero_idx])

        T = T_z = im_reshape = im = points = points_final = None  # free up memory
        self.depth_img = None
        self.semantic_img = None
        
        # create point cloud    
        self._createOpen3DCloud()
        return
    
    def showPointCloud(self):
        if not self._is_constructed:
            print("no reconstructed cloud")
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        o3d.visualization.draw_geometries([self._pcd, origin], mesh_show_wireframe=True) # visualize point cloud 
        return

    def savePointCloud(self, is_shown=False):
        if not self._is_constructed:
            print("save points failed, no reconstructed cloud!")

        print("save output files to: " + self._cfg.get_out_path())
        im_path = os.path.join(self._cfg.get_out_path(), "depth")
        if self._cfg.semantics:
            sem_path = os.path.join(self._cfg.get_out_path(), "semantic")
            
        if not os.path.exists(self._cfg.get_out_path()):
            os.makedirs(self._cfg.get_out_path())
            os.makedirs(im_path)
            # pre-create the folder for the mapping
            os.makedirs(os.path.join(*[self._cfg.get_out_path(), "maps", "cloud"]))
            os.makedirs(os.path.join(*[self._cfg.get_out_path(), "maps", "data"]))
            os.makedirs(os.path.join(*[self._cfg.get_out_path(), "maps", "params"]))
            
            if self._cfg.semantics:
                os.makedirs(sem_path)
                
        elif os.path.exists(im_path):
            # remove existing files
            for efile in os.listdir(im_path):
                os.remove(os.path.join(im_path, efile))
            if self._cfg.semantics and os.path.isdir(sem_path):
                for efile in os.listdir(sem_path):
                    os.remove(os.path.join(sem_path, efile))

        extrinsics_file_name = os.path.join(self._cfg.get_out_path(), "camera_extrinsic_ground_truth.txt")
        np.savetxt(extrinsics_file_name, self.extrinsics_list, delimiter=",")  # extrinsics are camera poses in robotics frame (x forward, y left, z up)

        # copy images
        for idx in range(len(self.extrinsics_list)):
            ipath_img_src = os.path.join(self._cfg.get_data_path(), "depth", str(idx).zfill(4) + ".png")
            ipath_img_dst = os.path.join(im_path, str(idx) + ".png")
            shutil.copyfile(ipath_img_src, ipath_img_dst)

            ipath_npy_src = os.path.join(self._cfg.get_data_path(), "depth", str(idx).zfill(4) + ".npy")
            if os.path.isfile(ipath_npy_src):
                ipath_npy_dst = os.path.join(im_path, str(idx) + ".npy")
                shutil.copyfile(ipath_npy_src, ipath_npy_dst)
                
            if self._cfg.semantics:
                sempath_src = os.path.join(self._cfg.get_data_path(), "semantics", str(idx).zfill(4) + ".png")
                sempath_dst = os.path.join(sem_path, str(idx) + ".png")
                shutil.copyfile(sempath_src, sempath_dst)
                
        # save intrinsic
        intrinsic_file_name = os.path.join(self._cfg.get_out_path(), "depth_intrinsic.txt")
        open(intrinsic_file_name, 'w').close()  # clear txt file
        fc = open(intrinsic_file_name, 'w')
        fc.writelines(str(self._intrinsic))
        fc.close()

        # save clouds
        o3d.io.write_point_cloud(os.path.join(self._cfg.get_out_path(), "cloud.ply"), self._pcd) # save point cloud

        print("saved point cloud to ply file.")
        return None

    @property
    def pcd(self):
        return self._pcd

    """helper functions"""
    
    def _createOpen3DCloud(self) -> None:
        print(f"creating open3d geometry point cloud with batch size of {self._cfg.point_cloud_batch_size}...")

        # init point cloud with first image
        pcd = o3d.geometry.PointCloud() # point size (n, 3)
        pcd.points = o3d.utility.Vector3dVector(self._points.pop(0))
        if self._cfg.semantics:
            pcd.colors = o3d.utility.Vector3dVector(self._sem_mapping.pop(0) / 255.0)    
        
        # repeat for all batches
        for batch_idx in tqdm(range(np.floor(len(self._points)/self._cfg.point_cloud_batch_size).astype(int)), desc ="Generate Open3D Point-Cloud"):
            pcd.points.extend(np.vstack(self._points[:self._cfg.point_cloud_batch_size]))
            del self._points[:self._cfg.point_cloud_batch_size] # delete first self._cfg.point_cloud_batch_size elements
            if self._cfg.semantics:
                pcd.colors.extend(np.vstack(self._sem_mapping[:self._cfg.point_cloud_batch_size]) / 255.0)
                del self._sem_mapping[:self._cfg.point_cloud_batch_size] # delete first self._cfg.point_cloud_batch_size elements
        
        # add last batch
        pcd.points.extend(np.vstack(self._points))
        self._points = None
        if self._cfg.semantics:
            pcd.colors.extend(np.vstack(self._sem_mapping) / 255.0)
            self._sem_mapping = None
        
        # apply downsampling
        self._pcd = pcd.voxel_down_sample(self._cfg.voxel_size)
        
        # update flag
        self._is_constructed = True
        print("construction completed.")
        return
    
    def _read_extrinsic(self) -> None:
        extrinsic_path = os.path.join(self._cfg.get_data_path(), "camera_extrinsic.txt")
        self.extrinsics_list = np.loadtxt(extrinsic_path, delimiter=',')
        return 

    def _read_intrinsic(self) -> None:
        intrinsic_path = os.path.join(self._cfg.get_data_path(), "intrinsics.txt")
        P = np.loadtxt(intrinsic_path, delimiter=",")  # assumes ROS P matrix
        self._intrinsic = list(P)
        self.K = P.reshape(3, 4)[:3, :3]
        return 

    def _load_depth_images(self) -> np.ndarray:
        # get path to images
        dir_path = os.path.join(self._cfg.get_data_path(), "depth")
        # init
        img_array = np.zeros((self._end_idx-self._start_idx, 360, 640))  # int(self.K[1, 2]*2), int(self.K[0, 2]*2)))  # assumes camera center in the middle of image plane
        # repeat for all further images
        for idx in range(self._start_idx, self._end_idx):
            if os.path.isfile(os.path.join(dir_path, str(idx).zfill(4) + ".npy")):
                img_array[idx-self._start_idx] =  np.load(os.path.join(dir_path, str(idx).zfill(4) + ".npy")) / self._cfg.depth_scale
            else:
                img_path = os.path.join(dir_path, str(idx).zfill(4) + ".png")
                img_array[idx-self._start_idx] = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) / self._cfg.depth_scale
            
            if self._cfg.max_depth:
                img_array[idx-self._start_idx] = np.clip(img_array[idx-self._start_idx], 0, self._cfg.max_depth)
        
        img_array[~np.isfinite(img_array)] = 0
        return img_array

    def _load_semantic_images(self) -> np.ndarray:
        # get path to images
        dir_path = os.path.join(self._cfg.get_data_path(), "semantics")
        # init
        img_array = np.zeros((self._end_idx-self._start_idx, int(self.K[1, 2]*2), int(self.K[0, 2]*2), 3))  # assumes camera center in the middle of image plane
        # repeat for all further images
        for idx in range(self._start_idx, self._end_idx):
            img_path = os.path.join(dir_path, str(idx).zfill(4) + ".png")
            img_array[idx-self._start_idx] = cv2.imread(img_path)
        return img_array
    
    def _computePixelTensor(self, x_nums, y_nums):
        T = np.zeros([3, x_nums, y_nums])
        for u in range(x_nums):
            for v in range(y_nums):
                T[:, u, v] = np.array([u, v, 1.0])
        return T


if __name__ == '__main__':
    cfg = ReconstructionCfg()
        
    # start depth reconstruction
    depth_constructor = DepthReconstruction(cfg)
    depth_constructor.depthMapReconstruction()

    depth_constructor.savePointCloud()
    depth_constructor.showPointCloud()

# EoF
