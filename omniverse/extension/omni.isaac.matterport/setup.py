# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Installation script for the 'omni.isaac.matterport' python package."""


from setuptools import find_namespace_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "trimesh",
    "PyQt5",
    "matplotlib>=3.5.0",
    "pandas",
]

# Installation operation
setup(
    name="omni-isaac-matterport",
    author="Pascal Roth",
    author_email="rothpa@ethz.ch",
    version="0.0.1",
    description="Extension to include Matterport 3D Datasets into Isaac (taken from https://niessner.github.io/Matterport/).",
    keywords=["robotics"],
    include_package_data=True,
    python_requires=">=3.11,<3.12",
    install_requires=INSTALL_REQUIRES,
    packages=find_namespace_packages(include=["omni.*"]),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)

# EOF
