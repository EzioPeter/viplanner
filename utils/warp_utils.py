#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      warp utils
"""

# python
import torch
import warp as wp
from typing import Tuple, Union

def _raycast(
    mesh: wp.Mesh, 
    ray_starts_world: torch.Tensor,
    ray_directions_world: torch.Tensor,
    max_depth: Union[float, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs ray casting on the terrain mesh.

    Args:
        ray_starts_world (Torch.tensor): The starting position of the ray.
        ray_directions_world (Torch.tensor): The ray direction.

    Returns:
        [Torch.tensor]: The ray hit position. Returns float('inf') for missed hits.
    """
    # TODO: make function more memory efficient
    shape = ray_starts_world.shape
    ray_starts_world = ray_starts_world.reshape(-1,3).contiguous()
    ray_directions_world = ray_directions_world.reshape(-1, 3).contiguous()
    num_rays = len(ray_starts_world)
    if not isinstance(max_depth, torch.Tensor):
        max_depth = torch.ones(num_rays, device=ray_starts_world.device) * max_depth
    max_depth = max_depth.reshape(-1).contiguous()    
    
    # get pointers to arrays
    ray_starts_world_wp = wp.types.array(
        ptr=ray_starts_world.data_ptr(),
        dtype=wp.vec3,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=mesh.device,
    )
    ray_directions_world_wp = wp.types.array(
        ptr=ray_directions_world.data_ptr(),
        dtype=wp.vec3,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=mesh.device,
    )
    ray_hits_world = torch.zeros((num_rays, 3), device=ray_starts_world.device)
    ray_hits_world[:] = float("inf")
    ray_hits_world_wp = wp.types.array(
        ptr=ray_hits_world.data_ptr(),
        dtype=wp.vec3,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=mesh.device
    )
    ray_hit_depth = torch.zeros(num_rays, device=ray_starts_world.device)
    ray_hit_depth[:] = float("inf")
    ray_hit_depth_wp = wp.types.array(
        ptr=ray_hit_depth.data_ptr(),
        dtype=wp.float32,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=mesh.device
    )
    max_depth_wp = wp.types.array(
        ptr=max_depth.data_ptr(),
        dtype=wp.float32,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=mesh.device
    )        
    wp.launch(
        kernel=_raycast_kernel,
        dim=num_rays,
        inputs=[mesh.id, ray_starts_world_wp, ray_directions_world_wp, ray_hits_world_wp, ray_hit_depth_wp, max_depth_wp],
        device=mesh.device,
    )
    wp.synchronize()
    
    return ray_hits_world, ray_hit_depth

@wp.kernel
def _raycast_kernel(
    mesh: wp.uint64,
    ray_starts_world: wp.array(dtype=wp.vec3),
    ray_directions_world: wp.array(dtype=wp.vec3),
    ray_hits_world: wp.array(dtype=wp.vec3),
    ray_hit_depth: wp.array(dtype=wp.float32),
    max_depth: wp.array(dtype=wp.float32)
):

    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index
    # ray cast against the mesh
    if wp.mesh_query_ray(mesh, ray_starts_world[tid], ray_directions_world[tid], max_depth[tid], t, u, v, sign, n, f):
        ray_hits_world[tid] = ray_starts_world[tid] + t * ray_directions_world[tid]
        ray_hit_depth[tid] = t
    return