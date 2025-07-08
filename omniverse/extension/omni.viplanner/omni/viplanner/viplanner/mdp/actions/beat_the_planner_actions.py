from __future__ import annotations

import torch
from omni.isaac.lab.devices import Se2Gamepad
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils.configclass import configclass

from .navigation_actions import NavigationAction, NavigationActionCfg


# -- Navigation Action
class BeatThePlannerAction(NavigationAction):
    """Actions to navigate a robot by following some path."""

    cfg: BeatThePlannerActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: BeatThePlannerActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # initialize gamepad
        self.gamepad = Se2Gamepad()

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            # -- update command
            self._env.command_manager.compute(dt=self._low_level_step_dt)
            # -- overwrite one command with gamepad
            self._env.command_manager._terms["vel_command"].twist[self.cfg.gamepad_controlled_robot_id] = torch.tensor(
                self.gamepad.advance(), device=self._env.device
            ) * torch.tensor([1, -1, -1], device=self._env.device)
            # Get low level actions from low level policy
            self._low_level_actions[:] = self.low_level_policy(
                self._env.observation_manager.compute_group(group_name="policy")
            )
            # Process low level actions
            self.low_level_action_term.process_actions(self._low_level_actions)

        # Apply low level actions
        self.low_level_action_term.apply_actions()
        self._counter += 1


@configclass
class BeatThePlannerActionCfg(NavigationActionCfg):
    class_type: type[ActionTerm] = BeatThePlannerAction
    """ Class of the action term."""
    gamepad_controlled_robot_id: int = 1
