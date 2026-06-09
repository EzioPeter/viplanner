# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize a trained IsaacLab ANYmal-D velocity policy on flat ground."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from _bootstrap import add_local_extensions_to_pythonpath, append_local_extensions_to_kit_args


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "anymal_d_flat_teleop"


def _site_packages_dir() -> Path:
    return Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def add_isaaclab_source_paths() -> None:
    source_root = _site_packages_dir() / "isaaclab" / "source"
    for package in ("isaaclab_assets", "isaaclab_rl", "isaaclab_tasks"):
        path = source_root / package
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an IsaacLab ANYmal-D velocity policy on flat ground.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="RSL-RL checkpoint. Defaults to latest model_final.pt/model_*.pt under logs.",
    )
    parser.add_argument("--cmd_x", default=0.2, type=float, help="Fixed base-frame linear x command.")
    parser.add_argument("--cmd_y", default=0.0, type=float, help="Fixed base-frame linear y command.")
    parser.add_argument("--cmd_yaw", default=0.0, type=float, help="Fixed base-frame yaw-rate command.")
    parser.add_argument("--num_envs", default=1, type=int, help="Number of visualized environments.")
    parser.add_argument("--max_steps", default=0, type=int, help="Stop after this many env steps. 0 means run forever.")
    parser.add_argument("--debug_interval", default=60, type=int, help="Print status every N env steps. 0 disables.")
    parser.add_argument("--seed", default=1234, type=int, help="Environment seed.")
    parser.add_argument(
        "--deterministic_reset",
        action="store_true",
        help="Remove reset pose/joint randomization for cleaner visual inspection.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.kit_args = append_local_extensions_to_kit_args(args_cli.kit_args)
    return args_cli


add_local_extensions_to_pythonpath()
add_isaaclab_source_paths()

from isaaclab.app import AppLauncher


args_cli = parse_args()
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def find_latest_checkpoint() -> Path:
    candidates = [*DEFAULT_LOG_ROOT.glob("*/model_final.pt"), *DEFAULT_LOG_ROOT.glob("*/model_*.pt")]
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No RSL-RL checkpoint found under {DEFAULT_LOG_ROOT}. "
            "Pass --checkpoint /path/to/model.pt or train with train_anymal_d_velocity.py."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_policy(device: str, checkpoint: Path):
    import torch
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict

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
    return policy


def quat_wxyz_to_roll_pitch_deg(quat) -> tuple[float, float]:
    import math

    w = float(quat[0])
    x = float(quat[1])
    y = float(quat[2])
    z = float(quat[3])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    return math.degrees(roll), math.degrees(pitch)


def main() -> None:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg_PLAY
    from tensordict import TensorDict

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve() if args_cli.checkpoint else find_latest_checkpoint()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    env_cfg = AnymalDFlatEnvCfg_PLAY()
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_x, args_cli.cmd_x)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.cmd_y, args_cli.cmd_y)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.cmd_yaw, args_cli.cmd_yaw)

    if args_cli.deterministic_reset:
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
    policy = load_policy(env.device, checkpoint)
    command_term = env.command_manager._terms["base_velocity"]

    with torch.inference_mode():
        env.reset()
        policy_obs = TensorDict(
            {"policy": env.observation_manager.compute_group(group_name="policy")},
            batch_size=[env.num_envs],
        )
    env.sim.play()

    step_count = 0
    command_tensor = torch.tensor(
        [[args_cli.cmd_x, args_cli.cmd_y, args_cli.cmd_yaw]],
        dtype=command_term.vel_command_b.dtype,
        device=env.device,
    ).repeat(env.num_envs, 1)
    print(
        f"[INFO] Flat play ready: checkpoint={checkpoint}, "
        f"cmd=({args_cli.cmd_x:.3f},{args_cli.cmd_y:.3f},{args_cli.cmd_yaw:.3f}), "
        f"num_envs={env.num_envs}",
        flush=True,
    )

    try:
        while args_cli.max_steps <= 0 or step_count < args_cli.max_steps:
            if args_cli.max_steps <= 0 and not simulation_app.is_running():
                break
            if not env.sim.is_playing():
                env.sim.play()

            command_term.vel_command_b[:] = command_tensor
            command_term.is_standing_env[:] = False
            command_term.is_heading_env[:] = False

            with torch.inference_mode():
                actions = policy.act_inference(policy_obs)
                step_result = env.step(actions)
                policy_obs = TensorDict({"policy": step_result[0]["policy"]}, batch_size=[env.num_envs])

            if args_cli.debug_interval > 0 and step_count % args_cli.debug_interval == 0:
                robot = env.scene["robot"]
                pos = robot.data.root_pos_w[0].detach().cpu().tolist()
                root_vel_b = robot.data.root_lin_vel_b[0].detach().cpu().tolist()
                roll_deg, pitch_deg = quat_wxyz_to_roll_pitch_deg(robot.data.root_quat_w[0].detach().cpu().tolist())
                print(
                    f"[PLAY] step={step_count} base_vel=({root_vel_b[0]:.3f},"
                    f"{root_vel_b[1]:.3f},{root_vel_b[2]:.3f}) "
                    f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                    f"rpy=({roll_deg:.1f},{pitch_deg:.1f})",
                    flush=True,
                )
            step_count += 1
            if not args_cli.headless:
                time.sleep(0.0)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
