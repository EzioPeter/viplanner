"""
@author     Pascal Roth
@email      rothpa@ethz.ch

@brief      segments.ai config for own dataset labeling
"""

# python
import os
from dataclasses import dataclass


@dataclass
class SegmentsCfg:
    # API key
    api_key: str = "ee0a626ee7c160e6c841dcd59743b811bf25c774"
    # dataset
    dataset_name: str = "leggedrobotics/urban_navigation"
    version: str = "v0.5"
    # export parameters
    export_format: str = "coco-panoptic"
    export_dir: str = "zurich_own_new"

    @property
    def export_dir_path(self) -> str:
        return os.path.join(
            os.getenv(
                "EXPERIMENT_DIRECTORY",
                "/home/pascal/viplanner/imperative_learning/",
            ),
            "data",
            self.export_dir,
        )

    @property
    def export_file_path(self) -> str:
        return os.path.join(
            self.export_dir_path,
            "segments",
            self.dataset_name.replace("/", "_"),
        )
