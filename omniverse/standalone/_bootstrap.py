# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local bootstrap helpers for running ViPlanner against pip/uv Isaac Sim."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_DIR = REPO_ROOT / "omniverse" / "extension"
EXTENSION_PATHS = (
    EXTENSION_DIR / "omni.viplanner",
    EXTENSION_DIR / "omni.isaac.matterport",
    EXTENSION_DIR / "omni.waypoints",
)


def add_local_extensions_to_pythonpath() -> None:
    """Make this repo's Omniverse extension modules importable."""
    for path in reversed(EXTENSION_PATHS):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def append_local_extensions_to_kit_args(existing_args: str) -> str:
    """Add local extension search paths for Kit without forcing extension activation."""
    kit_args = [
        existing_args.strip(),
        *(f"--ext-folder={path}" for path in EXTENSION_PATHS if path.exists()),
    ]
    return " ".join(arg for arg in kit_args if arg)


def close_simulation_app(simulation_app) -> None:
    """Close Isaac Sim without waiting on expensive stage cleanup."""
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
