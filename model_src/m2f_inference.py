"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Mask2Former optimized inference script to be used in the ROS node
"""

# python
import time
import numpy as np
import torchvision.transforms as transforms

from detectron2.config import get_cfg, CfgNode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor

# ROS
import rospy

# viplanner-ros
from .coco_meta import get_class_for_id
from .viplanner_sem_meta import VIPlannerSemMetaHandler

# mask2former src
from .mask2former.mask2former import add_maskformer2_config

# viplanner src
from .viplanner.config import TrainCfg


class Mask2FormerInference:
    """Run Inference on Mask2Former model to estimate semantic segmentation"""

    debug: bool = False
    
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
        self._cfg_train = TrainCfg()
        
        # load model
        self.predictor = DefaultPredictor(self._cfg)

        # transforms
        self.transforms = transforms.Compose([
            transforms.Resize(self._cfg_train.img_input_size),
            transforms.ToTensor()]
        )
        
        # mapping from coco class id to viplanner class id and corresponding color 
        viplanner_meta = VIPlannerSemMetaHandler()
        coco_viplanner_cls_mapping = get_class_for_id()
        self.viplanner_sem_class_color_map = viplanner_meta.class_color
        self.coco_viplanner_color_mapping = {}
        for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
            self.coco_viplanner_color_mapping[coco_id] = viplanner_meta.class_color[viplanner_cls_name]
        
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
        panoptic_mask = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1], 3), dtype=np.uint8)
        for sinfo in seg_infos:
            try:
                panoptic_mask[panoptic_seg.cpu().numpy() == sinfo['id']] = self.coco_viplanner_color_mapping[sinfo['category_id']]
            except KeyError:
                rospy.loginfo(f"Category {sinfo['category_id']} not found in coco_viplanner_cls_mapping.")
                panoptic_mask[panoptic_seg.cpu().numpy() == sinfo['id']] = self.viplanner_sem_class_color_map['static']
        rospy.loginfo("Semantic Pred. time: {:.3f}s".format(time.time() - start))
        
        if self.debug:
            import matplotlib.pyplot as plt
            plt.imshow(panoptic_mask)
            plt.show()
        
        return panoptic_mask
        
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
