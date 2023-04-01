#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      warp utils
"""

# m2f
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from third_party.mask2former.mask2former import add_maskformer2_config
from third_party.mask2former.demo.predictor import VisualizationDemo


def load_m2f_demo(model_path, config_path):
    setup_logger(name="fvcore")
    logger = setup_logger()
    
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(["MODEL.WEIGHTS", model_path])
    cfg.freeze()
    
    print("Network expects images of type: ", cfg.INPUT.FORMAT)
    
    # create predictor
    demo = VisualizationDemo(cfg)
    
    return demo

# EoF
