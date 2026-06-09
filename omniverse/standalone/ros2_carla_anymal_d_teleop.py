# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate ANYmal-D in the CARLA map with ROS 2 ``geometry_msgs/Twist``.

This standalone script intentionally bypasses VIPlanner's learned visual planner.
By default it uses an IsaacLab/RSL-RL ANYmal-D velocity policy trained by
``train_anymal_d_velocity.py`` and feeds ROS 2 velocity commands directly into
the IsaacLab command term.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time

from _bootstrap import add_local_extensions_to_pythonpath, append_local_extensions_to_kit_args, close_simulation_app


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARLA_ASSET_DIR = REPO_ROOT / "assets" / "new_carla_export"
DEFAULT_ISAACLAB_RSL_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "anymal_d_flat_teleop"


def _prepend_env_path(name: str, paths: list[Path]) -> None:
    existing = [value for value in os.environ.get(name, "").split(os.pathsep) if value]
    new_values = [str(path) for path in paths if path.exists() and str(path) not in existing]
    if new_values:
        os.environ[name] = os.pathsep.join([*new_values, *existing])


def _prepend_env_path_in(env: dict[str, str], name: str, paths: list[Path]) -> None:
    existing = [value for value in env.get(name, "").split(os.pathsep) if value]
    new_values = [str(path) for path in paths if path.exists() and str(path) not in existing]
    if new_values:
        env[name] = os.pathsep.join([*new_values, *existing])


def _site_packages_dir() -> Path:
    return Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def _ros2_distro_root(ros_distro: str) -> Path:
    bridge_root = _site_packages_dir() / "isaacsim" / "exts" / "isaacsim.ros2.bridge"
    distro_root = bridge_root / ros_distro
    if not distro_root.exists():
        available = sorted(path.name for path in bridge_root.iterdir() if path.is_dir()) if bridge_root.exists() else []
        raise FileNotFoundError(
            f"ROS 2 distro '{ros_distro}' was not found under {bridge_root}. Available: {available}"
        )
    return distro_root


def ensure_bundled_ros2_process_env(ros_distro: str) -> None:
    """Restart once so ROS 2 shared libraries are visible to the dynamic loader."""
    distro_root = _ros2_distro_root(ros_distro)
    marker = f"{ros_distro}:{distro_root}"
    if os.environ.get("VIPLANNER_ROS2_ENV_READY") == marker:
        configure_bundled_ros2(ros_distro)
        return

    env = os.environ.copy()
    _prepend_env_path_in(env, "PYTHONPATH", [distro_root / "rclpy"])
    _prepend_env_path_in(env, "LD_LIBRARY_PATH", [distro_root / "lib"])
    _prepend_env_path_in(env, "AMENT_PREFIX_PATH", [distro_root])
    env.setdefault("ROS_DISTRO", ros_distro)
    env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    env["VIPLANNER_ROS2_ENV_READY"] = marker
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def add_isaaclab_source_paths() -> None:
    """Make IsaacLab's source-layout asset packages importable."""
    source_root = _site_packages_dir() / "isaaclab" / "source"
    for package in ("isaaclab_assets", "isaaclab_rl", "isaaclab_tasks"):
        path = source_root / package
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def configure_bundled_ros2(ros_distro: str) -> Path:
    """Expose IsaacSim's bundled ROS 2 Python packages to this uv process."""
    distro_root = _ros2_distro_root(ros_distro)
    rclpy_path = distro_root / "rclpy"
    lib_path = distro_root / "lib"
    _prepend_env_path("PYTHONPATH", [rclpy_path])
    _prepend_env_path("LD_LIBRARY_PATH", [lib_path])
    _prepend_env_path("AMENT_PREFIX_PATH", [distro_root])
    os.environ.setdefault("ROS_DISTRO", ros_distro)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    rclpy_path_str = str(rclpy_path)
    if rclpy_path.exists() and rclpy_path_str not in sys.path:
        sys.path.insert(0, rclpy_path_str)
    return distro_root


@dataclass
class TwistCommand:
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0
    stamp: float = 0.0


class Ros2TwistSubscriber:
    """Small rclpy wrapper that stores the latest Twist command."""

    def __init__(self, topic: str, ros_distro: str, status_topic: str):
        configure_bundled_ros2(ros_distro)

        import rclpy
        from geometry_msgs.msg import Twist
        from std_msgs.msg import String

        self._rclpy = rclpy
        self._status_msg_type = String
        self._command = TwistCommand(stamp=time.monotonic())
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("isaacsim_anymal_d_teleop")
        self._subscription = self._node.create_subscription(Twist, topic, self._callback, 10)
        self._status_pub = self._node.create_publisher(String, status_topic, 10)
        print(f"[INFO] ROS 2 subscriber ready: topic={topic}, status={status_topic}, distro={ros_distro}")

    @property
    def latest(self) -> TwistCommand:
        return self._command

    def spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def shutdown(self) -> None:
        self._node.destroy_subscription(self._subscription)
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()

    def publish_status(self, text: str) -> None:
        msg = self._status_msg_type()
        msg.data = text
        self._status_pub.publish(msg)

    def _callback(self, msg) -> None:
        self._command = TwistCommand(
            linear_x=float(msg.linear.x),
            linear_y=float(msg.linear.y),
            angular_z=float(msg.angular.z),
            stamp=time.monotonic(),
        )


class ZeroCommandSubscriber:
    """Fallback command source for smoke tests without ROS 2."""

    latest = TwistCommand(stamp=time.monotonic())

    def spin_once(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def publish_status(self, text: str) -> None:
        return None


class FixedCommandSource:
    """Command source used for deterministic smoke tests."""

    def __init__(self, linear_x: float, linear_y: float, angular_z: float):
        self.latest = TwistCommand(linear_x=linear_x, linear_y=linear_y, angular_z=angular_z, stamp=time.monotonic())

    def spin_once(self) -> None:
        self.latest.stamp = time.monotonic()

    def shutdown(self) -> None:
        return None

    def publish_status(self, text: str) -> None:
        return None


class AnymalDOpenLoopGait:
    """Convert body velocity commands into ANYmal-D joint-position targets."""

    def __init__(
        self,
        joint_names: list[str],
        default_joint_pos,
        *,
        max_linear_velocity: float,
        max_lateral_velocity: float,
        max_yaw_rate: float,
        gait_frequency: float,
    ):
        self._joint_names = joint_names
        self._default = default_joint_pos.clone()
        self._phase = 0.0
        self._max_linear_velocity = max(max_linear_velocity, 1e-6)
        self._max_lateral_velocity = max(max_lateral_velocity, 1e-6)
        self._max_yaw_rate = max(max_yaw_rate, 1e-6)
        self._gait_frequency = gait_frequency
        self._joint_ids = {name: joint_names.index(name) for name in joint_names}
        self._legs = ("LF", "RF", "LH", "RH")
        self._leg_phase = {"LF": 0.0, "RH": 0.0, "RF": math.pi, "LH": math.pi}
        self._front_legs = {"LF", "RF"}
        self._left_legs = {"LF", "LH"}

    def compute(self, command: TwistCommand, dt: float, command_age: float, timeout: float):
        import torch

        target = self._default.clone()
        if command_age > timeout:
            return target

        linear_x = max(-self._max_linear_velocity, min(self._max_linear_velocity, command.linear_x))
        linear_y = max(-self._max_lateral_velocity, min(self._max_lateral_velocity, command.linear_y))
        angular_z = max(-self._max_yaw_rate, min(self._max_yaw_rate, command.angular_z))

        forward = linear_x / self._max_linear_velocity
        lateral = linear_y / self._max_lateral_velocity
        yaw = angular_z / self._max_yaw_rate
        effort = min(1.0, max(abs(forward), 0.7 * abs(lateral), 0.7 * abs(yaw)))
        if effort < 0.04:
            return target

        self._phase = (self._phase + 2.0 * math.pi * self._gait_frequency * (0.35 + 0.65 * effort) * dt) % (
            2.0 * math.pi
        )
        stride = 0.24 * effort
        knee_lift = 0.30 * effort
        hip_abduction = 0.12 * lateral
        turn_abduction = 0.10 * yaw

        for leg in self._legs:
            phase = self._phase + self._leg_phase[leg]
            swing = math.sin(phase)
            lift = max(0.0, swing)
            front_sign = 1.0 if leg in self._front_legs else -1.0
            left_sign = 1.0 if leg in self._left_legs else -1.0
            turn_sign = left_sign * yaw

            haa = hip_abduction * left_sign + turn_abduction * turn_sign
            hfe = front_sign * stride * swing - 0.05 * lateral * left_sign
            kfe = -front_sign * knee_lift * lift

            target[0, self._joint_ids[f"{leg}_HAA"]] += haa
            target[0, self._joint_ids[f"{leg}_HFE"]] += hfe
            target[0, self._joint_ids[f"{leg}_KFE"]] += kfe

        return torch.clamp(target, min=-1.35, max=1.35)


class BaseTwistDrive:
    """Move the robot base from Twist commands while the legs animate."""

    def __init__(
        self,
        robot,
        *,
        mode: str,
        base_height: float | None,
        max_linear_velocity: float,
        max_lateral_velocity: float,
        max_yaw_rate: float,
    ):
        self._robot = robot
        self._mode = mode
        self._base_height = base_height
        self._max_linear_velocity = max(max_linear_velocity, 1e-6)
        self._max_lateral_velocity = max(max_lateral_velocity, 1e-6)
        self._max_yaw_rate = max(max_yaw_rate, 1e-6)
        self._pose = robot.data.root_pose_w.clone()

    def apply(self, command: TwistCommand, dt: float, command_age: float, timeout: float) -> TwistCommand:
        import torch

        if self._mode == "none":
            return command

        if command_age > timeout:
            linear_x = 0.0
            linear_y = 0.0
            angular_z = 0.0
        else:
            linear_x = max(-self._max_linear_velocity, min(self._max_linear_velocity, command.linear_x))
            linear_y = max(-self._max_lateral_velocity, min(self._max_lateral_velocity, command.linear_y))
            angular_z = max(-self._max_yaw_rate, min(self._max_yaw_rate, command.angular_z))

        yaw = quat_wxyz_to_yaw(self._pose[0, 3:7])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        velocity_world_x = cos_yaw * linear_x - sin_yaw * linear_y
        velocity_world_y = sin_yaw * linear_x + cos_yaw * linear_y

        root_velocity = self._robot.data.root_vel_w.clone()
        root_velocity[0, 0] = velocity_world_x
        root_velocity[0, 1] = velocity_world_y
        root_velocity[0, 5] = angular_z
        self._robot.write_root_velocity_to_sim(root_velocity)

        if self._mode == "kinematic":
            self._pose[0, 0] += velocity_world_x * dt
            self._pose[0, 1] += velocity_world_y * dt
            if self._base_height is not None:
                self._pose[0, 2] = self._base_height
            self._pose[0, 3:7] = torch.tensor(
                yaw_to_quat_wxyz(yaw + angular_z * dt),
                dtype=self._pose.dtype,
                device=self._pose.device,
            )
            self._robot.write_root_pose_to_sim(self._pose)

        return TwistCommand(linear_x=linear_x, linear_y=linear_y, angular_z=angular_z, stamp=command.stamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 teleop for ANYmal-D in the CARLA map.")
    parser.add_argument("--cmd_vel_topic", default="/cmd_vel", type=str, help="ROS 2 Twist topic to subscribe to.")
    parser.add_argument(
        "--status_topic",
        default="/anymal_teleop/status",
        type=str,
        help="ROS 2 std_msgs/String status topic published by IsaacSim.",
    )
    parser.add_argument("--ros_distro", default="jazzy", choices=["humble", "jazzy"], help="Bundled ROS 2 distro.")
    parser.add_argument(
        "--carla_asset_dir",
        default=str(Path(os.environ.get("VIPLANNER_CARLA_ASSET_DIR", DEFAULT_CARLA_ASSET_DIR))),
        type=str,
        help="Directory containing carla.usd.",
    )
    parser.add_argument("--spawn_x", default=130.0, type=float, help="ANYmal-D spawn x in CARLA world.")
    parser.add_argument("--spawn_y", default=116.0, type=float, help="ANYmal-D spawn y in CARLA world.")
    parser.add_argument("--spawn_z", default=0.62, type=float, help="ANYmal-D spawn z in CARLA world.")
    parser.add_argument("--spawn_yaw", default=-90.0, type=float, help="ANYmal-D initial yaw in degrees.")
    parser.add_argument("--max_linear_velocity", default=0.25, type=float, help="Clamp for Twist linear.x.")
    parser.add_argument("--max_lateral_velocity", default=0.25, type=float, help="Clamp for Twist linear.y.")
    parser.add_argument("--max_yaw_rate", default=0.35, type=float, help="Clamp for Twist angular.z.")
    parser.add_argument("--gait_frequency", default=1.6, type=float, help="Open-loop gait cycle frequency.")
    parser.add_argument(
        "--controller",
        default="rsl_d",
        choices=["rsl_d", "policy", "velocity"],
        help="Use the ANYmal-D RSL-RL velocity policy, the original path follower policy, or the simple fallback.",
    )
    parser.add_argument(
        "--robot",
        default="anymal_d",
        choices=["anymal_d", "anymal_c"],
        help="Robot asset to spawn. The bundled locomotion policy is trained for ANYmal-C.",
    )
    parser.add_argument("--policy_max_speed", default=0.25, type=float, help="Max speed used by the path follower.")
    parser.add_argument("--path_horizon", default=6.0, type=float, help="Seconds covered by the synthesized path.")
    parser.add_argument(
        "--min_path_distance",
        default=1.5,
        type=float,
        help="Minimum forward distance for the synthesized policy path when a command is active.",
    )
    parser.add_argument(
        "--lookahead_distance",
        default=1.0,
        type=float,
        help="Lookahead distance used by the policy path follower.",
    )
    parser.add_argument(
        "--base_drive",
        default="velocity",
        choices=["kinematic", "velocity", "none"],
        help="How /cmd_vel drives the base. 'velocity' keeps gravity/contact active.",
    )
    parser.add_argument(
        "--base_height",
        default=None,
        type=float,
        help="Fixed base z for kinematic mode only. Leave unset for normal physics.",
    )
    parser.add_argument("--debug_cmd", action="store_true", help="Print received /cmd_vel and robot position.")
    parser.add_argument("--debug_step_interval", default=100, type=int, help="Debug print interval in sim steps.")
    parser.add_argument("--status_interval", default=0.5, type=float, help="Seconds between ROS 2 status messages.")
    parser.add_argument("--fixed_cmd_x", default=None, type=float, help="Smoke-test fixed command for linear.x.")
    parser.add_argument("--fixed_cmd_y", default=0.0, type=float, help="Smoke-test fixed command for linear.y.")
    parser.add_argument("--fixed_cmd_yaw", default=0.0, type=float, help="Smoke-test fixed command for angular.z.")
    parser.add_argument(
        "--command_timeout",
        default=2.0,
        type=float,
        help="Seconds before stale commands become zero.",
    )
    parser.add_argument(
        "--rsl_checkpoint",
        default=None,
        type=str,
        help="IsaacLab/RSL-RL ANYmal-D checkpoint used by --controller rsl_d. Defaults to latest trained model.",
    )
    parser.add_argument("--max_steps", default=0, type=int, help="Optional smoke-test step limit; 0 means run forever.")
    parser.add_argument(
        "--disable_ros2",
        action="store_true",
        help="Run standing smoke test without ROS 2 subscription.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.kit_args = append_local_extensions_to_kit_args(args_cli.kit_args)
    return args_cli


add_local_extensions_to_pythonpath()

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--ros_distro", default="jazzy", choices=["humble", "jazzy"])
_pre_parser.add_argument("--disable_ros2", action="store_true")
_pre_args, _ = _pre_parser.parse_known_args()
if not _pre_args.disable_ros2:
    ensure_bundled_ros2_process_env(_pre_args.ros_distro)

from isaaclab.app import AppLauncher

args_cli = parse_args()
if args_cli.controller == "policy":
    args_cli.enable_cameras = True

# Configure ROS 2 before Isaac Sim starts loading extension libraries.
if not args_cli.disable_ros2:
    configure_bundled_ros2(args_cli.ros_distro)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def quat_wxyz_to_yaw(quat) -> float:
    w = float(quat[0])
    x = float(quat[1])
    y = float(quat[2])
    z = float(quat[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_to_roll_pitch_deg(quat) -> tuple[float, float]:
    w = float(quat[0])
    x = float(quat[1])
    y = float(quat[2])
    z = float(quat[3])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    return math.degrees(roll), math.degrees(pitch)


def twist_to_policy_path(command: TwistCommand, path_length: int, device: str, dtype, args: argparse.Namespace):
    """Convert a base-frame Twist into the path action consumed by NavigationAction."""
    import torch

    linear_x = max(-args.max_linear_velocity, min(args.max_linear_velocity, command.linear_x))
    angular_z = max(-args.max_yaw_rate, min(args.max_yaw_rate, command.angular_z))
    if abs(linear_x) < 0.03 and abs(angular_z) < 0.03:
        return torch.zeros((1, path_length * 3), device=device, dtype=dtype)

    horizon = max(args.path_horizon, 0.2)
    forward_distance = abs(linear_x) * horizon
    if abs(linear_x) > 0.03:
        forward_distance = max(forward_distance, args.min_path_distance)
    signed_speed = math.copysign(forward_distance / horizon, linear_x) if abs(linear_x) > 0.03 else 0.0
    times = torch.linspace(0.0, horizon, path_length, device=device, dtype=dtype)
    path = torch.zeros((path_length, 3), device=device, dtype=dtype)

    if abs(angular_z) < 1.0e-3:
        path[:, 0] = signed_speed * times
    else:
        forward_speed = signed_speed
        if abs(forward_speed) < 0.08:
            forward_speed = 0.12 if angular_z >= 0.0 else -0.12
        radius = forward_speed / angular_z
        yaw = angular_z * times
        path[:, 0] = radius * torch.sin(yaw)
        path[:, 1] = radius * (1.0 - torch.cos(yaw))

    return path.reshape(1, path_length * 3)


def build_command_source(args: argparse.Namespace):
    if args.fixed_cmd_x is not None:
        return FixedCommandSource(args.fixed_cmd_x, args.fixed_cmd_y, args.fixed_cmd_yaw)
    if args.disable_ros2:
        return ZeroCommandSubscriber()
    return Ros2TwistSubscriber(args.cmd_vel_topic, args.ros_distro, args.status_topic)


def clamp_twist_command(command: TwistCommand, args: argparse.Namespace) -> TwistCommand:
    linear_x = max(-args.max_linear_velocity, min(args.max_linear_velocity, command.linear_x))
    linear_y = max(-args.max_lateral_velocity, min(args.max_lateral_velocity, command.linear_y))
    angular_z = max(-args.max_yaw_rate, min(args.max_yaw_rate, command.angular_z))
    return TwistCommand(linear_x=linear_x, linear_y=linear_y, angular_z=angular_z, stamp=command.stamp)


def find_latest_isaaclab_rsl_checkpoint() -> Path:
    candidates = sorted(DEFAULT_ISAACLAB_RSL_ROOT.glob("*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No IsaacLab ANYmal-D checkpoint found under {DEFAULT_ISAACLAB_RSL_ROOT}. "
            "Train one with omniverse/standalone/train_anymal_d_velocity.py, "
            "or pass --rsl_checkpoint /path/to/model.pt."
        )
    return candidates[-1]


def load_rsl_anymal_d_policy(device: str):
    import torch
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict

    checkpoint = (
        Path(args_cli.rsl_checkpoint).expanduser().resolve()
        if args_cli.rsl_checkpoint is not None
        else find_latest_isaaclab_rsl_checkpoint()
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"ANYmal-D RSL-RL checkpoint not found: {checkpoint}")

    sample_obs = TensorDict({"policy": torch.zeros((1, 48), dtype=torch.float32, device=device)}, batch_size=[1])
    policy = ActorCritic(
        obs=sample_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=12,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
    ).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
    policy.eval()
    return policy, checkpoint


def run_rsl_d_teleop() -> None:
    add_isaaclab_source_paths()

    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.prims as prim_utils
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg_PLAY
    from omni.viplanner.utils import UnRealImporterCfg
    from pxr import UsdGeom
    from tensordict import TensorDict

    carla_usd = Path(args_cli.carla_asset_dir).expanduser().resolve() / "carla.usd"
    if not carla_usd.exists():
        raise FileNotFoundError(f"CARLA USD not found: {carla_usd}")

    env_cfg = AnymalDFlatEnvCfg_PLAY()
    env_cfg.seed = 1234
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 1.0
    env_cfg.scene.replicate_physics = False
    env_cfg.scene.robot.init_state.pos = (args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_z)
    env_cfg.scene.robot.init_state.rot = yaw_to_quat_wxyz(math.radians(args_cli.spawn_yaw))
    env_cfg.scene.terrain = UnRealImporterCfg(
        prim_path="/World/Carla",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        usd_path=str(carla_usd),
        groundplane=True,
        axis_up="Z",
    )
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-args_cli.max_linear_velocity, args_cli.max_linear_velocity)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-args_cli.max_lateral_velocity, args_cli.max_lateral_velocity)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-args_cli.max_yaw_rate, args_cli.max_yaw_rate)
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    env = ManagerBasedRLEnv(env_cfg)
    env.sim.set_camera_view(
        eye=[args_cli.spawn_x + 7.5, args_cli.spawn_y + 8.0, args_cli.spawn_z + 5.5],
        target=[args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_z + 0.5],
    )
    if prim_utils.is_prim_path_valid("/World/GroundPlane"):
        prim_utils.get_prim_at_path("/World/GroundPlane").GetAttribute("visibility").Set(UsdGeom.Tokens.invisible)

    policy, checkpoint = load_rsl_anymal_d_policy(env.device)
    command_source = build_command_source(args_cli)
    command_term = env.command_manager._terms["base_velocity"]

    with torch.inference_mode():
        env.reset()
        policy_obs = TensorDict(
            {"policy": env.observation_manager.compute_group(group_name="policy")},
            batch_size=[env.num_envs],
        )
    env.sim.play()

    step_count = 0
    last_debug_print = 0.0
    last_status_publish = 0.0
    print(
        f"[INFO] RSL-D teleop ready: checkpoint={checkpoint}, debug={args_cli.debug_cmd}, "
        f"max_steps={args_cli.max_steps}, fixed_cmd_x={args_cli.fixed_cmd_x}",
        flush=True,
    )

    try:
        while args_cli.max_steps <= 0 or step_count < args_cli.max_steps:
            if args_cli.max_steps <= 0 and not simulation_app.is_running():
                break
            command_source.spin_once()
            if not env.sim.is_playing():
                env.sim.play()

            now = time.monotonic()
            command_age = now - command_source.latest.stamp
            active_command = command_source.latest if command_age <= args_cli.command_timeout else TwistCommand(stamp=now)
            active_command = clamp_twist_command(active_command, args_cli)

            command_tensor = torch.tensor(
                [[active_command.linear_x, active_command.linear_y, active_command.angular_z]],
                dtype=command_term.vel_command_b.dtype,
                device=env.device,
            )
            command_term.vel_command_b[:] = command_tensor
            command_term.is_standing_env[:] = False
            command_term.is_heading_env[:] = False

            with torch.inference_mode():
                actions = policy.act_inference(policy_obs)
                step_result = env.step(actions)
                policy_obs = TensorDict(
                    {"policy": step_result[0]["policy"]},
                    batch_size=[env.num_envs],
                )

            debug_by_time = now - last_debug_print > 0.5
            debug_by_steps = args_cli.debug_step_interval > 0 and step_count % args_cli.debug_step_interval == 0
            should_report = (args_cli.debug_cmd and (debug_by_time or debug_by_steps)) or (
                args_cli.status_interval > 0.0 and now - last_status_publish > args_cli.status_interval
            )
            if should_report:
                robot = env.scene["robot"]
                pos = robot.data.root_pos_w[0].detach().cpu().tolist()
                roll_deg, pitch_deg = quat_wxyz_to_roll_pitch_deg(robot.data.root_quat_w[0].detach().cpu().tolist())
                root_vel_b = robot.data.root_lin_vel_b[0].detach().cpu().tolist()
                status = (
                    f"step={step_count} playing={env.sim.is_playing()} controller=rsl_d "
                    f"ros=({active_command.linear_x:.3f},{active_command.linear_y:.3f},"
                    f"{active_command.angular_z:.3f}) "
                    f"base_vel=({root_vel_b[0]:.3f},{root_vel_b[1]:.3f},{root_vel_b[2]:.3f}) "
                    f"age={command_age:.3f}s pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                    f"rpy=({roll_deg:.1f},{pitch_deg:.1f})"
                )
                command_source.publish_status(status)
                if args_cli.debug_cmd and (debug_by_time or debug_by_steps):
                    print(f"[CMD] {status}", flush=True)
                last_debug_print = now
                last_status_publish = now
            step_count += 1
    finally:
        command_source.shutdown()
        env.close()


def run_policy_teleop() -> None:
    add_isaaclab_source_paths()

    import isaacsim.core.utils.prims as prim_utils
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_assets.robots.anymal import ANYMAL_C_CFG, ANYMAL_D_CFG
    from omni.viplanner.config import ViPlannerCarlaCfg
    from pxr import UsdGeom

    env_cfg = ViPlannerCarlaCfg(seed=1234)
    robot_cfg = ANYMAL_D_CFG if args_cli.robot == "anymal_d" else ANYMAL_C_CFG
    env_cfg.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.robot.init_state.pos = (args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_z)
    env_cfg.scene.robot.init_state.rot = yaw_to_quat_wxyz(math.radians(args_cli.spawn_yaw))
    env_cfg.commands.vel_command.path_frame = "robot"
    env_cfg.commands.vel_command.maxSpeed = args_cli.policy_max_speed
    env_cfg.commands.vel_command.lookAheadDistance = args_cli.lookahead_distance
    env_cfg.commands.vel_command.debug_vis = True

    env = ManagerBasedRLEnv(env_cfg)
    if prim_utils.is_prim_path_valid("/World/GroundPlane"):
        prim_utils.get_prim_at_path("/World/GroundPlane").GetAttribute("visibility").Set(UsdGeom.Tokens.invisible)

    with torch.inference_mode():
        env.reset()
    env.sim.play()

    command_source = build_command_source(args_cli)
    path_term = env.action_manager._terms["paths"]
    path_length = path_term.cfg.path_length
    step_count = 0
    last_debug_print = 0.0
    last_status_publish = 0.0
    print(
        f"[INFO] Policy teleop ready: robot={args_cli.robot}, path_length={path_length}, "
        f"debug={args_cli.debug_cmd}, max_steps={args_cli.max_steps}, fixed_cmd_x={args_cli.fixed_cmd_x}, "
        f"policy={path_term.cfg.low_level_policy_file}",
        flush=True,
    )

    try:
        while args_cli.max_steps <= 0 or step_count < args_cli.max_steps:
            if args_cli.max_steps <= 0 and not simulation_app.is_running():
                break
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
            command_source.spin_once()
            if not env.sim.is_playing():
                env.sim.play()

            now = time.monotonic()
            command_age = now - command_source.latest.stamp
            active_command = command_source.latest if command_age <= args_cli.command_timeout else TwistCommand(stamp=now)
            action = twist_to_policy_path(
                active_command,
                path_length,
                env.device,
                env.scene["robot"].data.default_joint_pos.dtype,
                args_cli,
            )
            if args_cli.debug_cmd and step_count == 0:
                print(f"[CMD] entering policy loop action_norm={float(action.norm()):.3f}", flush=True)

            with torch.inference_mode():
                env.action_manager.process_action(action.to(env.device))
                for _ in range(env.cfg.decimation):
                    env._sim_step_counter += 1
                    env.action_manager.apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    if not args_cli.headless and env._sim_step_counter % env.cfg.sim.render_interval == 0:
                        env.sim.render()
                    env.scene.update(dt=env.physics_dt)
            if args_cli.debug_cmd and step_count == 0:
                print("[CMD] returned policy step", flush=True)

            debug_by_time = now - last_debug_print > 0.5
            debug_by_steps = args_cli.debug_step_interval > 0 and step_count % args_cli.debug_step_interval == 0
            should_report = (args_cli.debug_cmd and (debug_by_time or debug_by_steps)) or (
                args_cli.status_interval > 0.0 and now - last_status_publish > args_cli.status_interval
            )
            if should_report:
                robot = env.scene["robot"]
                pos = robot.data.root_pos_w[0].detach().cpu().tolist()
                roll_deg, pitch_deg = quat_wxyz_to_roll_pitch_deg(robot.data.root_quat_w[0].detach().cpu().tolist())
                low_cmd = env.command_manager.get_command("vel_command")[0].detach().cpu().tolist()
                status = (
                    f"step={step_count} playing={env.sim.is_playing()} "
                    f"ros=({active_command.linear_x:.3f},{active_command.linear_y:.3f},"
                    f"{active_command.angular_z:.3f}) "
                    f"low=({low_cmd[0]:.3f},{low_cmd[1]:.3f},{low_cmd[2]:.3f}) "
                    f"age={command_age:.3f}s pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                    f"rpy=({roll_deg:.1f},{pitch_deg:.1f})"
                )
                command_source.publish_status(status)
                if args_cli.debug_cmd and (debug_by_time or debug_by_steps):
                    print(f"[CMD] {status}", flush=True)
                last_debug_print = now
                last_status_publish = now
            step_count += 1
    finally:
        command_source.shutdown()


def main() -> None:
    if args_cli.controller == "rsl_d":
        run_rsl_d_teleop()
        return

    if args_cli.controller == "policy":
        run_policy_teleop()
        return

    add_isaaclab_source_paths()

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab_assets.robots.anymal import ANYMAL_D_CFG
    from omni.viplanner.utils import UnRealImporterCfg

    carla_usd = Path(args_cli.carla_asset_dir).expanduser().resolve() / "carla.usd"
    if not carla_usd.exists():
        raise FileNotFoundError(f"CARLA USD not found: {carla_usd}")

    @configclass
    class CarlaTeleopSceneCfg(InteractiveSceneCfg):
        terrain = UnRealImporterCfg(
            prim_path="/World/Carla",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            usd_path=str(carla_usd),
            groundplane=True,
            axis_up="Z",
        )

        robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.init_state.pos = (args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_z)
        robot.init_state.rot = yaw_to_quat_wxyz(math.radians(args_cli.spawn_yaw))

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=1500.0),
        )

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, render_interval=4))
    sim.set_camera_view(
        eye=[args_cli.spawn_x + 7.5, args_cli.spawn_y + 8.0, args_cli.spawn_z + 5.5],
        target=[args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_z],
    )
    scene = InteractiveScene(CarlaTeleopSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False))
    sim.reset()
    sim.play()
    print("[INFO] CARLA + ANYmal-D scene is ready.", flush=True)

    robot = scene["robot"]
    gait = AnymalDOpenLoopGait(
        robot.joint_names,
        robot.data.default_joint_pos,
        max_linear_velocity=args_cli.max_linear_velocity,
        max_lateral_velocity=args_cli.max_lateral_velocity,
        max_yaw_rate=args_cli.max_yaw_rate,
        gait_frequency=args_cli.gait_frequency,
    )
    base_drive = BaseTwistDrive(
        robot,
        mode=args_cli.base_drive,
        base_height=args_cli.base_height,
        max_linear_velocity=args_cli.max_linear_velocity,
        max_lateral_velocity=args_cli.max_lateral_velocity,
        max_yaw_rate=args_cli.max_yaw_rate,
    )
    command_source = build_command_source(args_cli)

    sim_dt = sim.get_physics_dt()
    step_count = 0
    last_debug_print = 0.0
    try:
        while simulation_app.is_running():
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
            if sim.is_stopped():
                break
            command_source.spin_once()
            if not sim.is_playing():
                sim.step(render=not args_cli.headless)
                continue

            now = time.monotonic()
            active_command = base_drive.apply(
                command_source.latest,
                sim_dt,
                now - command_source.latest.stamp,
                args_cli.command_timeout,
            )
            joint_targets = gait.compute(
                active_command,
                sim_dt,
                now - active_command.stamp,
                args_cli.command_timeout,
            )
            robot.set_joint_position_target(joint_targets)
            scene.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            scene.update(sim_dt)
            debug_by_time = now - last_debug_print > 0.5
            debug_by_steps = args_cli.debug_step_interval > 0 and step_count % args_cli.debug_step_interval == 0
            if args_cli.debug_cmd and (debug_by_time or debug_by_steps):
                pos = robot.data.root_pos_w[0].detach().cpu().tolist()
                print(
                    "[CMD] "
                    f"x={active_command.linear_x:.3f} y={active_command.linear_y:.3f} "
                    f"yaw={active_command.angular_z:.3f} age={now - active_command.stamp:.3f}s "
                    f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                    flush=True,
                )
                last_debug_print = now
            step_count += 1
    finally:
        command_source.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_simulation_app(simulation_app)
