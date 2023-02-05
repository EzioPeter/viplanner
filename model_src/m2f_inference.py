"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Mask2Former optimized inference script to be used in the ROS node
"""

# python
import numpy as np
import torchvision.transforms as transforms

from detectron2.config import get_cfg, CfgNode
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor

# mask2former src
from mask2former import add_maskformer2_config


class Mask2FormerInference:
    """Run Inference on Mask2Former model to estimate semantic segmentation"""

    def __init__(
        self,
        config_file="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        model_weights="model_final.pth",
    ) -> None:
        
        # set arguments
        self._config_file = config_file
        self._model_weights = ['MODEL.WEIGHTS', model_weights]
        
        # setup config
        self._cfg = self._setup_cfg()

        # load model
        self.predictor = DefaultPredictor(self._cfg)

        # transforms
        self.transforms = transforms.Compose([
            transforms.Resize(tuple(self._crop_size)),
            transforms.ToTensor()]
        )
        return
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict semantic segmentation from image

        Args:
            image (np.ndarray): image to be processed
        """
        # Convert image to OpenCV BGR format
        image = image[:, :, ::-1]
        
        return self.predictor(image)
        
    """Helper functions"""
    
    def _setup_cfg(self) -> CfgNode:
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(self._config_file)
        cfg.merge_from_list(self._model_weights)
        cfg.freeze()
        return cfg

# EoF
