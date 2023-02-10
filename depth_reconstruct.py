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
import scipy.spatial.transform as tf
from tqdm import tqdm

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
        - depth  (either png and/ or npy)
            - xxxx.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
            - xxxx.npy  (arrays should be named with 4 digits, e.g. 0000.npy, 0001.npy, etc.)
        - semantics (optional)
            - xxxx.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
    
    when both depth and semantic images are available, then define sem_suffic and depth_suffix in ReconstructionCfg to differentiate between the two with the following structure:
    
    - env_name
        - camera_extrinsic{depth_suffix}.txt  (format: x y z qx qy qz qw)
        - camera_extrinsic{sem_suffix}.txt  (format: x y z qx qy qz qw)
        - intrinsics.txt (expects ROS CameraInfo format --> P-Matrix) (contains both intrinsics for depth and semantic images)
        - depth (either png and/ or npy)
            - xxxx{depth_suffix}.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
            - xxxx{depth_suffix}.npy  (arrays should be named with 4 digits, e.g. 0000.npy, 0001.npy, etc.)
        - semantics (optional)
            - xxxx{sem_suffix}.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
    
    in the case of high resolution depth images for the reconstruction, the following additional directory is expected:
        - depth_high_res (either png and/ or npy)
            - xxxx{depth_suffix}.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
            - xxxx{depth_suffix}.npy  (arrays should be named with 4 digits, e.g. 0000.npy, 0001.npy, etc.)
    """
    
    debug = False
    
    def __init__(self, cfg: ReconstructionCfg):
        # get config
        self._cfg: ReconstructionCfg = cfg
        # read camera params and odom
        self.K_depth: np.ndarray = None
        self.K_sem: np.ndarray = None
        self._read_intrinsic()
        self.extrinsics_depth: np.ndarray = None
        self.extrinsics_sem: np.ndarray = None
        self._read_extrinsic()
        # control flag if point-cloud has been loaded
        self._is_constructed =  False
        
        # variables
        self._points: np.ndarray = None
        self._pcd: o3d.geometry.PointCloud = None
        
        print("Ready to read depth data.")
        return None

    # public methods
    def depth_reconstruction(self):
        # identify start and end image idx for the reconstruction
        N = len(self.extrinsics_depth)
        if self._cfg.max_images:
            self._start_idx = self._cfg.start_idx
            self._end_idx = min(self._cfg.start_idx + self._cfg.max_images, N)
        else:
            self._start_idx = 0
            self._end_idx = N
        
        # load images
        self.depth_img = self._load_depth_images()
        
        # if semantic images are used, adjust images (only use the pixels present in both images)
        # if self._cfg.semantics:        
        #     self._compute_overlay_pixel()
        print(f"total number of images for reconstruction: {self.depth_img.shape[0]}")
            
        pixels = self._computePixelTensor()
        
        # init lists
        self._points = []
        if self._cfg.semantics:
            self._sem_mapping = []
            
        for img_idx in tqdm(range(self.depth_img.shape[0]), desc="Reconstructing 3D Points"):
            im = self.depth_img[img_idx]
            extrinsics = self.extrinsics_depth[img_idx+self._start_idx]
            
            # project points in world frame
            rot = tf.Rotation.from_quat(extrinsics[3:]).as_matrix()
            points = im.reshape(-1, 1) * (rot @ pixels.T).T
            # filter points with 0 depth --> otherwise obstacles at camera position
            non_zero_idx = np.where(points.any(axis=1))[0]
            
            points_final = points[non_zero_idx] + extrinsics[:3]       
                 
            if self._cfg.semantics:
                sem_annotation, filter_idx = self._get_semantic_image(points_final, img_idx)
                self._points.append(points_final[filter_idx])
                self._sem_mapping.append(sem_annotation)
            else:
                self._points.append(points_final)

        T = T_z = im_reshape = im = points = points_final = None  # free up memory
        self.depth_img = None
        self.semantic_img = None
        
        # create point cloud    
        self._createOpen3DCloud()
        return
    
    def show_pcd(self):
        if not self._is_constructed:
            print("no reconstructed cloud")
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        o3d.visualization.draw_geometries([self._pcd, origin], mesh_show_wireframe=True) # visualize point cloud 
        return

    def save_pcd(self):
        if not self._is_constructed:
            print("save points failed, no reconstructed cloud!")
        
        print("save output files to: " + os.path.join(self._cfg.data_dir, self._cfg.env))
        
        # pre-create the folder for the mapping
        os.makedirs(os.path.join(os.path.join(self._cfg.data_dir, self._cfg.env), "maps", "cloud"), exist_ok=True)
        os.makedirs(os.path.join(os.path.join(self._cfg.data_dir, self._cfg.env), "maps", "data"), exist_ok=True)
        os.makedirs(os.path.join(os.path.join(self._cfg.data_dir, self._cfg.env), "maps", "params"), exist_ok=True)
         
        # save clouds
        o3d.io.write_point_cloud(os.path.join(self._cfg.data_dir, self._cfg.env, "cloud.ply"), self._pcd) # save point cloud

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
        extrinsic_path = os.path.join(self._cfg.get_data_path(), "camera_extrinsic" + self._cfg.depth_suffix + ".txt")
        self.extrinsics_depth = np.loadtxt(extrinsic_path, delimiter=',')
        if self._cfg.semantics:
            extrinsic_path = os.path.join(self._cfg.get_data_path(), "camera_extrinsic" + self._cfg.sem_suffix + ".txt")
            self.extrinsics_sem = np.loadtxt(extrinsic_path, delimiter=',')        
        return 

    def _read_intrinsic(self) -> None:
        intrinsic_path = os.path.join(self._cfg.get_data_path(), "intrinsics.txt")
        P = np.loadtxt(intrinsic_path, delimiter=",")  # assumes ROS P matrix
        self._intrinsic = list(P)
        if self._cfg.semantics:
            self.K_depth = P[0].reshape(3, 4)[:3, :3]
            self.K_sem = P[1].reshape(3, 4)[:3, :3]
        else:
            self.K_depth = P.reshape(3, 4)[:3, :3]
        return 

    def _load_depth_images(self) -> np.ndarray:
        # get path to images
        if self._cfg.high_res_depth:
            dir_path = os.path.join(self._cfg.get_data_path(), "depth_high_res")
            res_factor = self._cfg.res_factor
        else:
            dir_path = os.path.join(self._cfg.get_data_path(), "depth")
            res_factor = 1
        # init
        img_array = np.zeros((self._end_idx-self._start_idx, int(self.K_depth[1, 2])*2*res_factor, int(self.K_depth[0, 2])*2*res_factor))  # assumes camera center in the middle of image plane
        # repeat for all further images
        for idx in range(self._start_idx, self._end_idx):
            if os.path.isfile(os.path.join(dir_path, str(idx).zfill(4) + self._cfg.depth_suffix + ".npy")):
                img_array[idx-self._start_idx] =  np.load(os.path.join(dir_path, str(idx).zfill(4) + self._cfg.depth_suffix + ".npy")) / self._cfg.depth_scale
            else:
                img_path = os.path.join(dir_path, str(idx).zfill(4) + self._cfg.depth_suffix + ".png")
                img_array[idx-self._start_idx] = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH) / self._cfg.depth_scale
        
        img_array[~np.isfinite(img_array)] = 0
        return img_array
      
    def _computePixelTensor(self):  
        # get image plane mesh grid
        pix_u = np.arange(0, self.depth_img.shape[2])
        pix_v = np.arange(0, self.depth_img.shape[1])
        grid = np.meshgrid(pix_u, pix_v)
        pixels = np.vstack(list(map(np.ravel, grid))).T
        pixels = np.hstack(
            [pixels, np.ones((len(pixels), 1))]
        )  # add ones for 3D coordinates 
        
        # transform to camera frame
        k_inv = np.linalg.inv(self.K_depth)
        pix_cam_frame = np.matmul(k_inv, pixels.T)
        # reorder to be in "robotics" axis order (x forward, y left, z up)
        return pix_cam_frame[[2, 0, 1], :].T  * np.array([1, -1, -1])          

    def _get_semantic_image(self, points, idx):
        # load semantic image and pose
        img_path = os.path.join(self._cfg.get_data_path(), "semantics", str(self._start_idx + idx).zfill(4) + self._cfg.sem_suffix + ".png")
        sem_image = cv2.imread(img_path)
        pose_sem   = self.extrinsics_sem[idx + self._cfg.start_idx]
        # transform points to semantic camera frame
        points_sem_cam_frame = (tf.Rotation.from_quat(pose_sem[3:]).as_matrix().T @ (points - pose_sem[:3]).T).T
        # normalize points
        points_sem_cam_frame_norm = points_sem_cam_frame / points_sem_cam_frame[:, 0][:, np.newaxis]
        # reorder points be camera convention (z-forward)
        points_sem_cam_frame_norm = points_sem_cam_frame_norm[:, [1, 2, 0]]  * np.array([-1, -1, 1])
        # transform points to pixel coordinates
        pixels = (self.K_sem @ points_sem_cam_frame_norm.T).T
        # filter points outside of image
        filter_idx = (pixels[:, 0] >= 0) & (pixels[:, 0] < sem_image.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < sem_image.shape[0])
        # get semantic annotation
        sem_annotation = sem_image[pixels[filter_idx, 1].astype(int), pixels[filter_idx, 0].astype(int)]
        
        return sem_annotation, filter_idx


if __name__ == '__main__':
    cfg = ReconstructionCfg()

    # start depth reconstruction
    depth_constructor = DepthReconstruction(cfg)
    depth_constructor.depth_reconstruction()

    depth_constructor.save_pcd()
    depth_constructor.show_pcd()

# EoF
