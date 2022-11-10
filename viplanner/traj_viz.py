
import os
import tf
import copy
import torch
import numpy as np
import pypose as pp
import open3d as o3d
import open3d.visualization.rendering as rendering


class TrajViz:
    def __init__(self, root_path, is_sim, cameraTilt=0.0):
        self.camera_tilt = cameraTilt
        if is_sim:
            file_name = "sim" + "_intrinsic.txt"
        else:
            file_name = "robot" + "_intrinsic.txt"
        intrinsic_path = os.path.join(*[root_path, file_name])
        self.SetCamera(intrinsic_path)
        return None

    def TransformPoints(self, odom, points):
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(batch_size, num_p, device=points.device, requires_grad=points.requires_grad)
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps

    def SetCamera(self, intrinsic_path, img_width=640, img_height=360):
        with open(intrinsic_path) as f:
            lines = f.readlines()
            elems = np.fromstring(lines[0][1:-2], dtype=float, sep=', ')
        K = np.array(elems).reshape(3, 4)
        self.camera = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, K[0,0], K[1,1], K[0,2], K[1,2])
        return

    def VizImages(self, preds, waypoints, odom, goal, fear, images, visual_offset=0.35, mesh_size=0.3):
        batch_size, _, _ = waypoints.shape
        preds_ws = self.TransformPoints(odom, preds)
        wp_ws = self.TransformPoints(odom, waypoints)
        if goal.shape[-1] != 7:
            pp_goal = pp.identity_SE3(batch_size, device=goal.device)
            pp_goal.tensor()[:, 0:3] = goal
            goal = pp_goal.tensor()
        goal_ws  = pp.SE3(odom) @ pp.SE3(goal)
        # convert to positions
        preds_ws = preds_ws.tensor()[:, :, 0:3].cpu().detach().numpy()
        wp_ws = wp_ws.tensor()[:, :, 0:3].cpu().detach().numpy()
        goal_ws  = goal_ws.tensor()[:, 0:3].cpu().detach().numpy()
        # adjust height
        preds_ws[:, :, 2] = preds_ws[:, :, 2] - visual_offset
        wp_ws[:, :, 2] = wp_ws[:, :, 2] - visual_offset
        goal_ws[:, 2] = goal_ws[:, 2] - visual_offset

        # set materia shader
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 0.3]
        mtl.shader = "defaultUnlit"
        # set meshes
        small_sphere      = o3d.geometry.TriangleMesh.create_sphere(mesh_size/20.0) # trajectory points
        mesh_sphere       = o3d.geometry.TriangleMesh.create_sphere(mesh_size/5.0) # successful predict points
        mesh_sphere_fear  = o3d.geometry.TriangleMesh.create_sphere(mesh_size/5.0) # unsuccessful predict points
        mesh_box          = o3d.geometry.TriangleMesh.create_box(mesh_size, mesh_size, mesh_size) # end points
        # set colors
        small_sphere.paint_uniform_color([0.99, 0.2, 0.1]) # green
        mesh_sphere.paint_uniform_color([0.4, 1.0, 0.1])
        mesh_sphere_fear.paint_uniform_color([1.0, 0.64, 0.0])
        mesh_box.paint_uniform_color([1.0, 0.64, 0.1])

        # init open3D render
        render = rendering.OffscreenRenderer(self.camera.width, self.camera.height)
        render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA

        # wp_start_idx = int(waypoints.shape[1] / preds.shape[1])
        wp_start_idx = 1
        cv_img_list = []

        for i in range(batch_size):
            # add geometries
            gp = goal_ws[i, :]
            # add goal marker
            goal_mesh = copy.deepcopy(mesh_box).translate((gp[0]-mesh_size/2.0, gp[1]-mesh_size/2.0, gp[2]-mesh_size/2.0))
            render.scene.add_geometry("goal_mesh", goal_mesh, mtl)
            # add predictions
            for j in range(preds_ws[i, :, :].shape[0]):
                kp = preds_ws[i, j, :]
                if fear[i, :] > 0.5:
                    kp_mesh = copy.deepcopy(mesh_sphere_fear).translate((kp[0], kp[1], kp[2]))
                else:
                    kp_mesh = copy.deepcopy(mesh_sphere).translate((kp[0], kp[1], kp[2]))
                render.scene.add_geometry("keypose"+str(j), kp_mesh, mtl)
            # add trajectory
            for k in range(wp_start_idx, wp_ws[i, :, :].shape[0]):
                wp = wp_ws[i, k, :]
                wp_mesh = copy.deepcopy(small_sphere).translate((wp[0], wp[1], wp[2]))
                render.scene.add_geometry("waypoint"+str(k), wp_mesh, mtl)
            # set cameras
            self.CameraLookAtPose(odom[i, :], render, self.camera_tilt)
            # project to image
            img_o3d = np.asarray(render.render_to_image())
            mask = (img_o3d < 10).all(axis=2)
            # Attach image
            c_img = images[i, :, :].expand(3, -1, -1)
            c_img = c_img.cpu().detach().numpy().transpose(1, 2, 0)
            c_img = (c_img * 255 / np.max(c_img)).astype('uint8')
            img_o3d[mask, :] = c_img[mask, :]
            cv_img_list.append(img_o3d)
            # clear render geometry
            render.scene.clear_geometry()
        return cv_img_list

    def CameraLookAtPose(self, odom, render, tilt):
        unit_vec = pp.identity_SE3(device=odom.device)
        unit_vec.tensor()[0] = 1.0
        tilt_vec = [0, 0, 0]
        tilt_vec.extend(list(tf.transformations.quaternion_from_euler(0.0, tilt, 0.0)))
        tilt_vec = torch.tensor(tilt_vec, device=odom.device, dtype=odom.dtype)
        target_pose = pp.SE3(odom) @ pp.SE3(tilt_vec) @ unit_vec
        camera_up = [0, 0, 1]  # camera orientation
        eye = odom[0:3].cpu().detach().numpy()
        target = target_pose.tensor()[0:3].cpu().detach().numpy()
        render.scene.camera.look_at(target, eye, camera_up)
        return
    