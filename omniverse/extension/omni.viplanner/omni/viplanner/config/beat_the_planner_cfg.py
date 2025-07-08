import math
import os

import omni.viplanner.viplanner.mdp as mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .carla_cfg import ViPlannerCarlaCfg
from .matterport_cfg import ViPlannerMatterportCfg
from .warehouse_cfg import ViPlannerWarehouseCfg

from ..viplanner.mdp.actions.beat_the_planner_actions import BeatThePlannerActionCfg

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    paths = BeatThePlannerActionCfg(
        asset_name="robot",
        low_level_decimation=4,
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        low_level_policy_file=os.path.join(ISAACLAB_NUCLEUS_DIR, "Policies", "ANYmal-C", "HeightScan", "policy.pt"),
        gamepad_controlled_robot_id=1,
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (-0.5, -0.5), "yaw": (-math.pi / 2, -math.pi / 2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# Environment configuration
##


@configclass
class BeatThePlannerMatterportCfg(ViPlannerMatterportCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    # managers
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2

        # reduce the number of rendering steps to make it real-time
        self.sim.render_interval = 16


@configclass
class BeatThePlannerCarlaCfg(ViPlannerCarlaCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    # managers
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2

        # reduce the number of rendering steps to make it real-time
        self.sim.render_interval = 16


@configclass
class BeatThePlannerWarehouseCfg(ViPlannerWarehouseCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    # managers
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2

        # reduce the number of rendering steps to make it real-time
        self.sim.render_interval = 16
