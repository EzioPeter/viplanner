
import os
import math
from scipy import ndimage
from tsdf_map import TSDF_Map
import open3d as o3d
import numpy as np
from scipy.ndimage import gaussian_filter
    

class TSDF_Creator:
    def __init__(self, root_path, voxel_size, robot_height, clear_dist=1.0, offset_z=0):
        self.root_path = root_path
        self.is_map_ready = False
        self.clear_dist = clear_dist
        self.voxel_size = voxel_size
        self.offset_z = offset_z
        self.robot_height = robot_height
        self.obs_pcd = o3d.geometry.PointCloud()
        self.free_pcd = o3d.geometry.PointCloud()

    def UpdatePCDwithPs(self, P_obs, P_free, is_downsample=False):
        self.obs_pcd.points  = o3d.utility.Vector3dVector(P_obs)
        self.free_pcd.points = o3d.utility.Vector3dVector(P_free)
        if is_downsample:
            self.obs_pcd  = self.obs_pcd.voxel_down_sample(self.voxel_size)
            self.free_pcd = self.free_pcd.voxel_down_sample(self.voxel_size * 0.85)

        self.obs_points   = np.asarray(self.obs_pcd.points)
        self.free_points  = np.asarray(self.free_pcd.points)
        print("number of obs points: %d, free points: %d"%(self.obs_points.shape[0], self.free_points.shape[0]))

    def ReadPointFromFile(self, file_name, is_filter=True):
        file_path = os.path.join(self.root_path, file_name)
        pcd_load = o3d.io.read_point_cloud(file_path)
        obs_p, free_p = self.TerrainAnalysis(np.asarray(pcd_load.points))
        self.UpdatePCDwithPs(obs_p, free_p, is_downsample=True)
        self.UpdateMapParams()
        if is_filter:
            obs_p  = self.FilterCloud(self.obs_points,  num_nbs=50, std_ratio=0.5)
            # free_p = self.FilterCloud(self.free_points, num_nbs=50, std_ratio=2.0)
            self.UpdatePCDwithPs(obs_p, free_p)
        return

    def AddAdditionalGround(self, file_name):
        file_path = os.path.join(self.root_path, file_name)
        pcd_load = o3d.io.read_point_cloud(file_path)
        _, free_p = self.TerrainAnalysis(np.asarray(pcd_load.points))
        # add the previous free points
        free_p = np.concatenate((self.free_points, free_p), axis=0)
        # change free pcd
        self.free_pcd.points = o3d.utility.Vector3dVector(free_p)
        # voxel downsample 
        self.free_pcd = self.free_pcd.voxel_down_sample(self.voxel_size * 0.85)
        self.free_points = np.asarray(self.free_pcd.points)
        print("Add additional free points, free points: %d"%(self.free_points.shape[0]))


    def TerrainAnalysis(self, input_points, ground_height=0.25):
        obs_points = np.zeros(input_points.shape)
        free_poins = np.zeros(input_points.shape)
        obs_idx = 0
        free_idx = 0
        # naive approach with z values
        for p in input_points:
            p_height = p[2] + self.offset_z
            if (p_height > ground_height * 1.2) and (p_height < self.robot_height * 1.5): # remove ground and ceiling
                obs_points[obs_idx, :] = p
                obs_idx = obs_idx + 1
            elif p_height < ground_height and p_height > -ground_height:
                free_poins[free_idx, :] = p
                free_idx = free_idx + 1
        return obs_points[:obs_idx, :], free_poins[:free_idx, :]

    def UpdateMapParams(self):
        if (self.obs_points.shape[0] == 0):
            print("No points received.")
            return
        max_x, max_y, _ = np.amax(self.obs_points, axis=0) + self.clear_dist
        min_x, min_y, _ = np.amin(self.obs_points, axis=0) - self.clear_dist
        
        self.num_x = np.ceil((max_x - min_x) / self.voxel_size / 10).astype(int) * 10
        self.num_y = np.ceil((max_y - min_y) / self.voxel_size / 10).astype(int) * 10
        self.start_x = (max_x + min_x) / 2.0 - self.num_x / 2.0 * self.voxel_size
        self.start_y = (max_y + min_y) / 2.0 - self.num_y / 2.0 * self.voxel_size

        print("tsdf map initialized, with size: %d, %d" %(self.num_x, self.num_y))
        self.is_map_ready = True
        return

    def CreateTSDFMap(self, sigma_expand=1.0, sigma_smooth=2.0):
        if not self.is_map_ready:
            print("create tsdf map fails, no points received.")
            return
        free_map = np.ones([self.num_x, self.num_y])
        obs_map = np.zeros([self.num_x, self.num_y])
        free_I = self.IndexArrayOfPs(self.free_points)
        obs_I = self.IndexArrayOfPs(self.obs_points)
        # create free place map
        thred = 1e-1
        for i in obs_I:
            obs_map[i[0], i[1]] = 1.0
        obs_map = gaussian_filter(obs_map, sigma=sigma_expand)
        for i in free_I:
            if i[0] < self.num_x and i[1] < self.num_y:
                free_map[i[0], i[1]] = 0
        free_map = gaussian_filter(free_map, sigma=sigma_expand)
        free_map[free_map < 0.5] = 0
        # assign obstacles
        free_map[obs_map > thred] = 1.0

        print("occupancy map generation completed.")
        # Distance Transform
        tsdf_array = ndimage.distance_transform_edt(free_map)
        tsdf_array[tsdf_array > 0.0]  = np.log(tsdf_array[tsdf_array > 0.0] + math.e)
        tsdf_array = gaussian_filter(tsdf_array, sigma=sigma_smooth)

        viz_points = np.concatenate((self.obs_points, self.free_points), axis=0)

        # TODO: Using true terrain analysis module
        ground_array = np.ones([self.num_x, self.num_y]) * 0.0
        return [tsdf_array, viz_points, ground_array], [self.start_x, self.start_y], [self.voxel_size, self.clear_dist]

    def IndexArrayOfPs(self, points):
        I = points[:, :2] - np.array([self.start_x, self.start_y])
        I = (np.round(I / self.voxel_size)).astype(int)
        return I

    def FilterCloud(self, points, num_nbs=100, std_ratio=1.0):
        # Filter outlier in self.obs_points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        cl, _ = pcd.remove_statistical_outlier(nb_neighbors=num_nbs, std_ratio=std_ratio)
        return np.asarray(cl.points)
    
    def VizCloud(self, pcd):
        o3d.visualization.draw_geometries([pcd]) # visualize point cloud 


if __name__ == '__main__':
    # empty
    is_lidar = False
    env_name = "2n8kARJN3HM"
    root_path = os.path.join("/home/pascal/SemNav/env/matterport/data_pc", env_name)
    map_name = "tsdf1"
    print("current root path", root_path)
    offset_z = 0.0

    print("============== Creating tsdf Map from cloud =================")
    if is_lidar:
        tsdf_creator = TSDF_Creator(root_path, voxel_size=0.1, offset_z=offset_z, robot_height=0.7)
        tsdf_creator.ReadPointFromFile("scan_cloud.ply")
        tsdf_creator.AddAdditionalGround("cloud.ply")
    else:
        tsdf_creator = TSDF_Creator(root_path, voxel_size=0.1, offset_z=offset_z, robot_height=0.7)
        tsdf_creator.ReadPointFromFile("cloud.ply")

    data, coord, params = tsdf_creator.CreateTSDFMap()
    tsdf_creator.VizCloud(tsdf_creator.obs_pcd)

    print("============== Generate tsdf Map =================")
    tsdf_map = TSDF_Map()
    tsdf_map.DirectLoadMap(data, coord, params)
    tsdf_map.SaveTSDFMap(root_path, map_name)
    # tsdf_map.ShowTSDFMap(cost_map=True)
    
    o3d.visualization.draw_geometries([tsdf_creator.obs_pcd, tsdf_map.pcd_tsdf])
    # print("============== Load tsdf Map =================")
    # tsdf_map_loader = TSDF_Map()
    # tsdf_map_loader.ReadTSDFMap(root_path, map_name)
    # tsdf_map_loader.ShowTSDFMap()






