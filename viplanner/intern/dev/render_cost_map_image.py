import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import PIL

from viplanner.cost_maps import CostMapPCD


def save_img_as_pdf(pcd, pc_path, img_name):
    # init open3D render
    mtl = rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    mtl.shader = "defaultUnlit"

    render = rendering.OffscreenRenderer(1920, 1080)
    render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA
    render.scene.add_geometry("pcd", pcd, mtl)

    camera_up = [0, 0, 1]  # camera orientation
    render.scene.camera.look_at([200, 150, 0], [150, 80, 70], camera_up)
    # print(render.scene.camera.get_model_matrix())

    img_o3d = np.asarray(render.render_to_image())

    plt.imshow(img_o3d)
    plt.show()

    pil_image = PIL.Image.fromarray(img_o3d)
    pil_image.save(pc_path + f"/{img_name}.pdf", quality=100)


def assign_color_to_pcd(pcd):
    points = np.asarray(pcd.points)
    colors = np.zeros((len(points), 3))

    # Define a custom colormap from blue to yellow to red
    n_bins = 100  # Number of bins in the colormap
    cm = mpl.colormaps["jet"].resampled(n_bins)
    # Assign colors based on height
    min_height = np.min(points[:, 2])
    max_height = np.max(points[:, 2])

    normalized_height = (points[:, 2] - min_height) / (max_height - min_height)
    colors = cm(normalized_height)[:, :3]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# point cloud
pc_path = "/home/pascal/viplanner/imperative_learning/data/town01_cam_mount_train"
img_name = "town01_pc"
pcd = o3d.io.read_point_cloud(pc_path + "/cloud.ply")
save_img_as_pdf(pcd, pc_path, img_name)
pcd = None

# cost map semantic
pc_path = "/home/pascal/viplanner/imperative_learning/data/town01_cam_mount_train"
map = CostMapPCD.ReadTSDFMap(pc_path, "cost_map_sem_sharpend")
pcd = assign_color_to_pcd(map.pcd_tsdf)
save_img_as_pdf(pcd, pc_path, "town01_sem")
map = pcd = None

# cost map tsdf
pc_path = "/home/pascal/viplanner/imperative_learning/data/town01_cam_mount_train"
map = CostMapPCD.ReadTSDFMap(pc_path, "tsdf")
pcd = assign_color_to_pcd(map.pcd_tsdf)
save_img_as_pdf(pcd, pc_path, "town01_depth")
map = pcd = None
