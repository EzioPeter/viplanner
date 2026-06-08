# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from .base_cfg import ViPlannerBaseCfg

##
# Pre-defined configs
##
# isort: off
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


def _mat(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.65)


def _static_box(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    semantic_class: str,
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=_mat(color),
            semantic_tags=[("class", semantic_class)],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


##
# Scene definition
##


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """A lightweight local obstacle map with clean primitive collisions."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(24.0, 24.0),
            color=(0.10, 0.12, 0.12),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            semantic_tags=[("class", "floor")],
        ),
    )

    wall_left = _static_box("/World/Obstacles/WallLeft", (0.25, 18.0, 1.2), (-4.5, 0.0, 0.6), "wall", (0.35, 0.36, 0.38))
    wall_right = _static_box("/World/Obstacles/WallRight", (0.25, 18.0, 1.2), (4.5, 0.0, 0.6), "wall", (0.35, 0.36, 0.38))
    wall_back = _static_box("/World/Obstacles/WallBack", (9.0, 0.25, 1.2), (0.0, -8.5, 0.6), "wall", (0.35, 0.36, 0.38))
    wall_front = _static_box("/World/Obstacles/WallFront", (9.0, 0.25, 1.2), (0.0, 8.5, 0.6), "wall", (0.35, 0.36, 0.38))

    crate_1 = _static_box("/World/Obstacles/Crate1", (1.4, 1.4, 1.0), (-1.7, 3.0, 0.5), "furniture", (0.65, 0.22, 0.18))
    crate_2 = _static_box("/World/Obstacles/Crate2", (1.6, 1.2, 1.0), (1.6, 1.0, 0.5), "furniture", (0.70, 0.45, 0.18))
    crate_3 = _static_box("/World/Obstacles/Crate3", (1.2, 1.8, 1.0), (-1.2, -1.8, 0.5), "furniture", (0.22, 0.42, 0.72))
    crate_4 = _static_box("/World/Obstacles/Crate4", (1.5, 1.2, 1.0), (1.9, -4.0, 0.5), "furniture", (0.52, 0.30, 0.66))
    pillar_1 = _static_box("/World/Obstacles/Pillar1", (0.7, 0.7, 1.8), (-3.0, -5.2, 0.9), "wall", (0.48, 0.50, 0.52))
    pillar_2 = _static_box("/World/Obstacles/Pillar2", (0.7, 0.7, 1.8), (3.0, 4.5, 0.9), "wall", (0.48, 0.50, 0.52))

    # robots
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 6.8, 0.6)
    robot.init_state.rot = (0.0, 0.0, 0.0, -1.0)

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/GroundPlane"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, debug_vis=False)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=1600.0),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/fill_light",
        spawn=sim_utils.SphereLightCfg(color=(1.0, 0.95, 0.85), intensity=2500.0),
    )
    fill_light.init_state.pos = (0.0, 1.5, 5.0)

    # cameras
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/depth_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(-0.5, 0.5, -0.5, 0.5)),
        spawn=sim_utils.PinholeCameraCfg(),
        width=848,
        height=480,
        data_types=["distance_to_image_plane"],
    )
    semantic_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/semantic_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(-0.5, 0.5, -0.5, 0.5)),
        spawn=sim_utils.PinholeCameraCfg(),
        width=1280,
        height=720,
        data_types=["semantic_segmentation", "rgb"],
        colorize_semantic_segmentation=False,
    )


##
# Environment configuration
##


@configclass
class ViPlannerObstaclesCfg(ViPlannerBaseCfg):
    """Configuration for a lightweight primitive obstacle map."""

    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (0.0, 10.5, 7.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
