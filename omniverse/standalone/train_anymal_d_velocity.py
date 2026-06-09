# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an IsaacLab ANYmal-D velocity policy for ROS 2 teleoperation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

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
    parser = argparse.ArgumentParser(description="Train IsaacLab ANYmal-D flat velocity policy.")
    parser.add_argument("--num_envs", default=1024, type=int, help="Parallel IsaacLab environments.")
    parser.add_argument("--max_iterations", default=300, type=int, help="PPO training iterations.")
    parser.add_argument("--save_interval", default=50, type=int, help="Checkpoint interval in iterations.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--run_name", default="ros2_teleop", type=str, help="Suffix for the run directory.")
    parser.add_argument("--log_root", default=str(DEFAULT_LOG_ROOT), type=str, help="Directory for RSL-RL logs.")
    parser.add_argument(
        "--num_steps_per_env",
        default=None,
        type=int,
        help="Override PPO rollout length. Leave unset for the official IsaacLab value.",
    )
    parser.add_argument(
        "--smoke_random_steps",
        default=0,
        type=int,
        help="Run this many random environment steps and exit without PPO training.",
    )
    parser.add_argument(
        "--disable_obs_corruption",
        action="store_true",
        help="Disable observation noise during training for a cleaner first smoke model.",
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


def main() -> None:
    import os

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg import (
        AnymalDFlatPPORunnerCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = AnymalDFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_obs_corruption:
        env_cfg.observations.policy.enable_corruption = False

    agent_cfg = AnymalDFlatPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.experiment_name = "anymal_d_flat_teleop"
    agent_cfg.run_name = args_cli.run_name
    agent_cfg.obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir_name = f"{timestamp}_{args_cli.run_name}" if args_cli.run_name else timestamp
    log_dir = Path(args_cli.log_root).expanduser().resolve() / run_dir_name
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Training IsaacLab ANYmal-D velocity policy")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs} iterations={agent_cfg.max_iterations} device={agent_cfg.device}")
    print(f"[INFO] num_steps_per_env={agent_cfg.num_steps_per_env}")
    print(f"[INFO] log_dir={log_dir}", flush=True)

    print("[INFO] creating env", flush=True)
    env = ManagerBasedRLEnv(env_cfg)
    print("[INFO] wrapping env", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.smoke_random_steps > 0:
        import torch

        print(f"[INFO] running random-action smoke steps: {args_cli.smoke_random_steps}", flush=True)
        obs, extras = env.reset()
        print(f"[INFO] reset ok: obs_shape={tuple(obs.shape)} extras_keys={list(extras.keys())}", flush=True)
        with torch.inference_mode():
            for step in range(args_cli.smoke_random_steps):
                actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
                obs, rewards, dones, infos = env.step(actions)
                print(
                    f"[INFO] smoke step {step + 1}/{args_cli.smoke_random_steps}: "
                    f"reward_mean={float(rewards.mean()):.4f} dones={int(dones.sum())}",
                    flush=True,
                )
        env.close()
        return

    print("[INFO] creating RSL-RL runner", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)

    print("[INFO] starting PPO learn", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    final_checkpoint = log_dir / "model_final.pt"
    runner.save(str(final_checkpoint))
    export_dir = log_dir / "exported"
    export_policy_as_jit(runner.alg.policy, None, path=str(export_dir), filename="policy.pt")
    print(f"[INFO] saved checkpoint: {final_checkpoint}")
    print(f"[INFO] exported jit policy: {export_dir / 'policy.pt'}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
