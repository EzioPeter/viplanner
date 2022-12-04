#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      config class for reconstruction and cost maps
"""

# python
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class ReconstructionCfg:
    """
    Arguments for 3D reconstruction using depth maps
    """
    # directory where the environment with the depth (and semantic) images is located
    data_dir: str = "/home/pascal/SemNav/env/matterport/data_domains"
    # directory where the reconstructed 3D map is saved
    out_dir: str = "/home/pascal/SemNav/env/matterport/data_pc"
    # environment name
    env: str = "2n8kARJN3HM"

    # reconstruction parameters
    voxel_size: float = 0.04
    start_idx: int = 0  # start index for reconstruction
    max_images: Optional[int] = None  # maximum number of images to reconstruct, fi None, all images are used

    # semantic reconstruction
    semantics: bool = False

    """ Internal functions """    
    def get_data_path(self) -> str:
        return os.path.join(self.data_dir, self.env)
    
    def get_out_path(self) -> str:
        return os.path.join(self.out_dir, self.env)

# EoF
