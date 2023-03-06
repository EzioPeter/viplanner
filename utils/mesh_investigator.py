#!/usr/bin python3

"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Investigate mesh for possible errors
"""

# python
import open3d as o3d
import trimesh
import numpy as np


def convex_decomposition(mesh):
    """
    Decompose mesh into convex parts

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to be decomposed

    Returns
    -------
    list of trimesh.Trimesh
        List of convex parts
    """
    # decompose mesh into convex parts
    convex_parts = mesh.convex_decomposition()
    return convex_parts

def load_mesh(mesh_path):
    """
    Load mesh from file

    Parameters
    ----------
    mesh_path : str
        Path to mesh file

    Returns
    -------
    trimesh.Trimesh
        Loaded mesh
    """
    # mesh = o3d.io.read_triangle_mesh(mesh_path, True)
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
    # o3d.visualization.draw_geometries([mesh, origin])    
    
    mesh_tri = trimesh.load_mesh(mesh_path, force_mesh=True)
    mesh = trimesh.util.concatenate(
    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
        for g in mesh_tri.geometry.values()))
    
    o3d_mesh = mesh.as_open3d
    
    open3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
    o3d.visualization.draw_geometries([o3d_mesh, origin])    
    
    scene = trimesh.Scene(mesh)
    scene.show()
    
    return mesh_tri


if __name__ == "__main__":
    mesh_path = "/home/pascal/SemNav/env/town01_fbx/Town01.obj"
    mesh = load_mesh(mesh_path)
    

    
    convex_parts = convex_decomposition(mesh)
    print(f"Mesh consists of {len(convex_parts)} convex parts.")
    

          