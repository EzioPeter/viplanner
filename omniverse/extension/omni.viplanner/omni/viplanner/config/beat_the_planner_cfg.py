import math
import os

import omni.viplanner.viplanner.mdp as mdp
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

from .carla_cfg import ViPlannerCarlaCfg
from .matterport_cfg import ViPlannerMatterportCfg
from .warehouse_cfg import ViPlannerWarehouseCfg

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    paths = mdp.BeatThePlannerActionCfg(
        asset_name="robot",
        low_level_decimation=4,
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        low_level_policy_file=os.path.join(ISAAC_ORBIT_NUCLEUS_DIR, "Policies", "ANYmal-C", "policy.pt"),
        gamepad_controlled_robot_id=1,
    )


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    reset_base = RandTerm(
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

    reset_robot_joints = RandTerm(
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

    # Scene settings
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    # managers
    randomization: RandomizationCfg = RandomizationCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2


@configclass
class BeatThePlannerCarlaCfg(ViPlannerCarlaCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    # managers
    randomization: RandomizationCfg = RandomizationCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2


@configclass
class BeatThePlannerWarehouseCfg(ViPlannerWarehouseCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    # managers
    randomization: RandomizationCfg = RandomizationCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
