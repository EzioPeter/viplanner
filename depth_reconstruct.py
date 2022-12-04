#!/usr/bin python3
import os
import cv2
import PIL
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
from ip_basic.ip_basic.depth_map_utils import fill_in_multiscale

class DepthReconstruction:
    def __init__(self, data_path, out_path, start_id, iters, voxel_size, is_max_iter=True):
        self._data_path = data_path
        self._out_path = out_path
        self._voxel_size = voxel_size
        self._is_max_iter = is_max_iter
        intrinsic_path = os.path.join(self._data_path, "intrinsics.txt")
        extrinsic_path = os.path.join(self._data_path, "camera_extrinsic.txt")
        # read camera params
        self._K = self._readIntrinsic(intrinsic_path)
        self._extrinsics_list = self._readExtrinsic(extrinsic_path)
        self._is_contruected =  False
        N = len(self._extrinsics_list)
        if self._is_max_iter:
            self._start_id = 0
            self._end_id = N
        else:
            self._start_id = start_id
            self._end_id = min(start_id + iters, N)

        print("Ready to read depth data.")
        return None

    # public methods
    def depthMapReconstruction(self):
        self._im_arr_list = []
        im_path = os.path.join(self._data_path, "depth")
        for idx in range(self._start_id, self._end_id):
            path = os.path.join(im_path, str(idx).zfill(4) + ".png")
            im = cv2.imread(path, cv2.IMREAD_ANYDEPTH).T
            self._im_arr_list.append(im.T)
        print("total number of images for reconstruction: {}".format(len(self._im_arr_list)))
        x_nums, y_nums = self._im_arr_list[0].shape
        T = self._computePixelTensor(x_nums, y_nums)
        pixel_nums = x_nums * y_nums
        print("start reconstruction...")
        self._points = np.zeros([(self._end_id-self._start_id+1) * pixel_nums, 3])
        # 3D reconstraction from Depth 
        k_inv = np.linalg.inv(self._K[:3, :3])     
        for idx in range(len(self._im_arr_list)):
            im = self._im_arr_list[idx] / 1000
            extrinsics = self._extrinsics_list[idx+self._start_id]
            
            rot = R.from_quat(extrinsics[3:]).as_euler("xyz", degrees=True)
            rot_transformed = np.zeros_like(rot)
            rot_transformed[1] = -rot[2]  # rotation around the z axis in robotics frame
            rot_transformed[0] = rot[1]  # rotation around the y axis in robotics frame
            rot_mat = R.from_euler("xyz", rot_transformed, degrees=True).as_matrix()
            T_z = T.reshape(3, -1)
            T_z = T_z[[1, 0, 2], :]  # reorder to be in camera frame (z forward, x right, y down)
            
            im_reshape = im.reshape(-1, 1)
            points = im_reshape * (rot_mat.T @ k_inv @ T_z).T
            points = points[:, [2, 0, 1]] * np.array([[1, -1, -1]])  # reorder to be in "robotics" frame (x forward, y left, z up)
            
            # filter points with 0 depth --> otherwise obstacles at camera position
            non_zero_idx = np.where(points.any(axis=1))[0]
            
            points_final = points[non_zero_idx] + extrinsics[:3]
            self._points[idx*pixel_nums: (idx)*pixel_nums + len(non_zero_idx), :] = points_final
            
        print("creating open3d geometry point cloud...")
        self._pcd = self._createOpen3DCloud(self._points, self._voxel_size)
        self._is_contruected = True
        print("construction completed.")
        
    def showPointCloud(self):
        if not self._is_contruected:
            print("no reconstructed cloud")
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        o3d.visualization.draw_geometries([self._pcd, origin], mesh_show_wireframe=True) # visualize point cloud 
        return

    def imageDilation(self, img, scale=1000.0, is_shown=False, is_rotated=False):
        if img.dtype.name == 'uint16':
            img = np.float32(img / scale)
        img, _ = fill_in_multiscale(img)
        # DEBUG Visual Image
        if is_shown:
            if is_rotated:
                c_img = np.array(Image.fromarray(img).transpose(PIL.Image.ROTATE_180))
            else:
                c_img = np.array(Image.fromarray(img))
            c_img = (c_img * 255 / np.max(c_img)).astype('uint8')
            cv2.imshow("Preview window", c_img)
            cv2.waitKey()
        img = img * scale
        return np.uint16(img)

    def savePointCloud(self, is_shown=False, is_rotated=False):
        if not self._is_contruected:
            print("save points failed, no reconstructed cloud!")

        print("save output files to: " + self._out_path)
        im_path = os.path.join(self._out_path, "depth")
        dila_path = os.path.join(self._out_path, "dilation")
        if not os.path.exists(self._out_path):
            os.makedirs(self._out_path)
            os.makedirs(im_path)
            os.makedirs(dila_path)
            # pre-create the folder for the mapping
            os.makedirs(os.path.join(*[self._out_path, "maps", "cloud"]))
            os.makedirs(os.path.join(*[self._out_path, "maps", "data"]))
            os.makedirs(os.path.join(*[self._out_path, "maps", "params"]))
        elif os.path.exists(im_path):
            # remove existing files
            for efile in os.listdir(im_path):
                os.remove(os.path.join(im_path, efile))

        extrinsics_file_name = os.path.join(self._out_path, "camera_extrinsic_ground_truth.txt")
        np.savetxt(extrinsics_file_name, self._extrinsics_list, delimiter=",")  # extrinsics are cameras in robotics frame (x forward, y left, z up)
        # open(extrinsics_file_name, 'w').close()  # clear txt file
        # extrinsics_file = open(extrinsics_file_name, 'w')
        # self._avg_z = 0.0
        for idx in range(len(self._im_arr_list)):
            ipath = os.path.join(im_path, str(idx) + ".png")
            dPath = os.path.join(dila_path, str(idx) + ".png")
            process_img = self._im_arr_list[idx]  #NOTE: removed the transpose
            cv2.imwrite(ipath, process_img)
            # Add image Dialation
            cv2.imwrite(dPath, self.imageDilation(process_img, is_shown=is_shown, is_rotated=is_rotated))
            # extrinsics_line = self._extrinsics_list[idx+self._start_id]
            # self._avg_z = self._avg_z + extrinsics_line[2]
        #     extrinsics_file.writelines(str(extrinsics_line))
        #     extrinsics_file.write('\n')
        # extrinsics_file.close()
        # self._avg_z =  self._avg_z / len(self._im_arr_list)

        # save intrinsic
        intrinsic_file_name = os.path.join(self._out_path, "depth_intrinsic.txt")
        open(intrinsic_file_name, 'w').close()  # clear txt file
        fc = open(intrinsic_file_name, 'w')
        fc.writelines(str(self._intrinsic))
        fc.close()

        # save clouds
        o3d.io.write_point_cloud(os.path.join(self._out_path, "cloud.ply"), self._pcd) # save point cloud

        print("saved point cloud to ply file.")
        return None

    @property
    def points(self):
        return self._points

    @property
    def pcd(self):
        return self._pcd

    @property
    def avg_z(self):
        return self._avg_z

    # prive methods
    def _createOpen3DCloud(self, points, voxel_size):
        pcd = o3d.geometry.PointCloud() # point size (n, 3)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return pcd
    
    def _readExtrinsic(self, odom_path):
        extrinsics = np.loadtxt(odom_path, delimiter=',')
        self._avg_z = np.mean(extrinsics[:,2])
        return extrinsics

    def _readIntrinsic(self, file_path):
        P = np.loadtxt(file_path, delimiter=",")
        self._intrinsic = list(P)
        # K = P.reshape(3, 4)[:3, :3]
        K = np.concatenate((P.reshape(3, 4), np.array([[0, 0, 0, 1]])), axis=0)
        return K

    def _computePixelTensor(self, x_nums, y_nums):
        T = np.zeros([3, x_nums, y_nums])
        for u in range(x_nums):
            for v in range(y_nums):
                T[:, u, v] = np.array([u, v, 1.0])
        return T

if __name__ == '__main__':
    # empty
    voxel_size = 0.04
    env_name = "2n8kARJN3HM" 

    start_id = 0
    iters = 500
    is_max_iter = True
    raw_dir = "/home/pascal/SemNav/env/matterport/data_domains"
    out_dir = "/home/pascal/SemNav/env/matterport/data_pc"
    raw_path = os.path.join(raw_dir, env_name)
    out_path = os.path.join(out_dir, env_name)
        
    # start depth reconstruction
    depth_constructor = DepthReconstruction(raw_path, out_path, start_id, iters, voxel_size, is_max_iter)
    depth_constructor.depthMapReconstruction()

    depth_constructor.savePointCloud()
    depth_constructor.showPointCloud()


