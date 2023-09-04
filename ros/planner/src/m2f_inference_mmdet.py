"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Mask2Former optimized inference script to be used in the ROS node
"""

# python
import numpy as np
from mmdet.apis import init_detector, inference_detector

# ROS
import rospy

# viplanner-ros
from viplanner.config.coco_meta import get_class_for_id_mmdet
from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler


class Mask2FormerInference:
    """Run Inference on Mask2Former model to estimate semantic segmentation"""

    debug: bool = False
    
    def __init__(
        self,
        config_file="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        checkpoint_file="model_final.pth",
    ) -> None:

        # Build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        
        # mapping from coco class id to viplanner class id and corresponding color 
        viplanner_meta = VIPlannerSemMetaHandler()
        coco_viplanner_cls_mapping = get_class_for_id_mmdet(self.model.dataset_meta["classes"])
        self.viplanner_sem_class_color_map = viplanner_meta.class_color
        self.coco_viplanner_color_mapping = {}
        for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
            self.coco_viplanner_color_mapping[coco_id] = viplanner_meta.class_color[viplanner_cls_name]
        
        return
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict semantic segmentation from image

        Args:
            image (np.ndarray): image to be processed in BGR format
        """
                
        result = inference_detector(self.model, image)
        result = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()
        # create output
        panoptic_mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for curr_sem_class in np.unique(result):
            try:
                panoptic_mask[result == curr_sem_class] = self.coco_viplanner_color_mapping[curr_sem_class]
            except KeyError:
                rospy.logwarn(f"Category {curr_sem_class} not found in coco_viplanner_cls_mapping.")
                panoptic_mask[result == curr_sem_class] = self.viplanner_sem_class_color_map['static']
        
        if self.debug:
            import matplotlib.pyplot as plt
            plt.imshow(panoptic_mask)
            plt.show()
        
        return panoptic_mask

# EoF
