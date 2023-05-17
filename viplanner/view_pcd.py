import open3d as o3d

# Load the PCD file
pcd_file = "/home/pascal/anymal_ros/catkin_ws/src/open3d_slam/open3d_slam_ros/data/maps/map.pcd"
pcd = o3d.io.read_point_cloud(pcd_file)

# Display the PCD file
o3d.visualization.draw_geometries([pcd])