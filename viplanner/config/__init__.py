from .costmap_cfg import ReconstructionCfg, SemCostMapConfig, TsdfCostMapConfig, CostMapConfig, GeneralCostMapConfig
from .learning_cfg import TrainCfg, DataCfg
from .viplanner_sem_meta import VIPlannerSemMetaHandler, OBSTACLE_LOSS

from .semantic_cfg import SegmentsCfg, Mask2FormerCfg

__all__ = [
    # configs
    "ReconstructionCfg", 
    "SemCostMapConfig", 
    "TsdfCostMapConfig",
    "CostMapConfig",
    "GeneralCostMapConfig",
    "TrainCfg",
    "DataCfg",
    "SegmentsCfg",
    "Mask2FormerCfg",
    # mapping
    "VIPlannerSemMetaHandler",
    "OBSTACLE_LOSS",
]


try:
    from .coco_meta import get_class_for_id, _COCO_MAPPING_UNIQUE, _COCO_MAPPING
    __all__ += ["get_class_for_id", "_COCO_MAPPING_UNIQUE", "_COCO_MAPPING"]
except ModuleNotFoundError:
    print("[WARNING] COCO meta cannot be used due to missing detectron2, skipping")

# EoF
