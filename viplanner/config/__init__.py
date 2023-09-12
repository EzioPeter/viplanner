from .costmap_cfg import (
    CostMapConfig,
    GeneralCostMapConfig,
    ReconstructionCfg,
    SemCostMapConfig,
    TsdfCostMapConfig,
)
from .learning_cfg import DataCfg, TrainCfg
from .semantic_cfg import Mask2FormerCfg, SegmentsCfg
from .viplanner_sem_meta import OBSTACLE_LOSS, VIPlannerSemMetaHandler

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
    from .coco_meta import (
        _COCO_MAPPING,
        _COCO_MAPPING_UNIQUE,
        get_class_for_id,
    )

    __all__ += ["get_class_for_id", "_COCO_MAPPING_UNIQUE", "_COCO_MAPPING"]
except ModuleNotFoundError:
    print(
        "[WARNING] COCO meta cannot be used due to missing detectron2,"
        " skipping"
    )

# EoF
