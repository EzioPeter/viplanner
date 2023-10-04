import os

import cv2
import matplotlib as mpl
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

# set materia shader
mtl = o3d.visualization.rendering.MaterialRecord()
mtl.base_color = [1.0, 1.0, 1.0, 1.0]
mtl.shader = "defaultUnlit"


def plotter(pcd, path, transparent=False):
    # get median of pcd
    pcd_points = np.asarray(pcd.points)
    pcd_median = np.median(pcd_points, axis=0)

    # init open3D render
    render = rendering.OffscreenRenderer(1280, 700)
    if transparent:
        render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA
    else:
        render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
    camera_up = [0, 0, 1]  # camera orientation
    render.scene.camera.look_at(pcd_median, pcd_median + np.array([0, 0, 30]), camera_up)
    render.scene.add_geometry("pcd", pcd, mtl)

    # project to image
    img_o3d = np.asarray(render.render_to_image())
    pcd_dir, file_name = os.path.split(path)
    file_name, _ = os.path.splitext(file_name)

    if transparent:
        img_o3d_white = np.all(img_o3d > 230, axis=2)
        img_o3d[img_o3d_white] = [255, 255, 255]
        img_o3d_transparent = np.zeros((img_o3d.shape[0], img_o3d.shape[1], 4), dtype=np.uint8)
        img_o3d_transparent[:, :, :3] = img_o3d
        img_o3d_transparent[~img_o3d_white, 3] = 255
        img_o3d = img_o3d_transparent

    img_cv2 = cv2.cvtColor(img_o3d, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(os.path.join(pcd_dir, f"{file_name}.png"), img_cv2)


def plot_trajectories(pcd_path: str, odom_path: str):
    # plot pcd
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = assign_color_to_pcd(pcd)
    plotter(pcd, pcd_path)

    # load odom
    odom = np.loadtxt(odom_path)[:1500]

    # create point cloud from odom
    odom_pcd = o3d.geometry.PointCloud()
    odom_pcd.points = o3d.utility.Vector3dVector(odom[:, :3])
    odom_color = np.zeros((len(odom), 3))
    odom_color[:, 1] = 1.0
    odom_pcd.colors = o3d.utility.Vector3dVector(odom_color)
    plotter(odom_pcd, odom_path, True)

    # visualize both
    o3d.visualization.draw_geometries([pcd, odom_pcd])
    print("done")


def assign_color_to_pcd(pcd):
    points = np.asarray(pcd.points)
    colors = np.zeros((len(points), 3))

    # Define a custom colormap from blue to yellow to red
    n_bins = 100  # Number of bins in the colormap
    # cm = Colormap("jet", N=n_bins).res
    cm = mpl.colormaps["gray"].resampled(n_bins)
    # Assign colors based on height
    min_height = np.min(points[:, 2])
    max_height = np.max(points[:, 2])

    normalized_height = (points[:, 2] - min_height) / (max_height - min_height)
    colors = cm(normalized_height)[:, :3]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


if __name__ == "__main__":
    plot_trajectories(
        pcd_path="/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_stairs_door/map.pcd",
        odom_path="/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_stairs_door/odom_base.txt",
    )
