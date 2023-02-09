"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Mask2Former optimized inference script to be used in the ROS node
"""

# python
import time
import torch
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
        start = time.time()
        # get predictions
        predictions = self.predictor(image)
        panoptic_seg, seg_infos = predictions['panoptic_seg']
        
        # create output
        segments = torch.zeros(panoptic_seg.shape).cuda()
        for sinfo in seg_infos:
            segments[panoptic_seg == sinfo['id']] = sinfo['category_id']+1
        panoptic_mask = 255*torch.ones((panoptic_seg.shape[0], panoptic_seg.shape[1], 3)).cuda()
        panoptic_mask[..., 2] = segments
        print("Pred. + vis. time: {:.3f}s".format(time.time() - start))
        
        # TODO: colorize panoptic mask 
        return np.rot90(panoptic_mask.cpu().numpy(), k=3).astype(np.uint8)
        
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
