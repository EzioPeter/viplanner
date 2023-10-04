"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Config for mask2former as semantic segmentation model
"""

# python
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Mask2FormerCfg:
    # path to model config file
    config: str = "coco/panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml"
    # path to model weights file
    model: str = "coco/panoptic/swin/model_final_9fd0ae.pkl"  # 'm2f_overfit/model_zurich_70_0049999.pth' #
    # output directory
    output: str = "m2f_overfit"

    # training parameters
    num_gpus: int = 1
    batch_size: int = 1
    epochs: int = 40
    coco_data: str = "coco"
    coco_nb_images: Optional[int] = 5000  # None for all images
    resume: bool = True
    eval_only: bool = False
    machine_rank: int = 0
    num_machines: int = 1
    dist_url: str = "tcp://127.0.0.1:50152"
    use_sem_seg: bool = (
        False  # semantic segmentation evaluation  --> also requires instancs json file (not available yet)
    )

    @property
    def config_file(self) -> str:
        return os.path.join(
            os.getenv("EXPERIMENT_DIRECTORY", "/home/pascal/viplanner/sem_seg"),
            "m2f_model",
            self.config,
        )

    @property
    def model_file(self) -> str:
        return os.path.join(
            os.getenv("EXPERIMENT_DIRECTORY", "/home/pascal/viplanner/sem_seg"),
            "m2f_model",
            self.model,
        )

    @property
    def coco_data_path(self) -> str:
        return os.path.join(
            os.getenv(
                "EXPERIMENT_DIRECTORY",
                "/home/pascal/viplanner/imperative_learning",
            ),
            "data",
            self.coco_data,
        )

    @property
    def output_path(self) -> str:
        return os.path.join(
            os.getenv("EXPERIMENT_DIRECTORY", "/home/pascal/viplanner/sem_seg"),
            "m2f_model",
            self.output,
        )
