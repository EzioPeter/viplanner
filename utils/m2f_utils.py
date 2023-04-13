#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      warp utils
"""
# python
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

# viplanner
from config import VIPlannerSemMetaHandler, get_class_for_id

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


def m2f_run_on_folder(rgb_folder, m2f_model, m2f_config):
    # check folder
    assert os.path.isdir(rgb_folder), f"Folder {rgb_folder} does not exist!"
    parent_folder, _ = os.path.split(rgb_folder) 
    sem_folder = os.path.join(parent_folder, "semantics")   
    os.makedirs(sem_folder, exist_ok=True)
    
    # get m2f model
    demo = load_m2f_demo(m2f_model, m2f_config)
    
    # get mapping
    viplanner_meta = VIPlannerSemMetaHandler()
    coco_viplanner_cls_mapping = get_class_for_id()
    viplanner_sem_class_color_map = viplanner_meta.class_color
    coco_viplanner_color_mapping = {}
    for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
        coco_viplanner_color_mapping[coco_id] = viplanner_meta.class_color[viplanner_cls_name]
    
    # init plot to show predictions
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((1080, 1440, 3), dtype=np.uint8))

    # load all images from the folder
    img_list = os.listdir(rgb_folder)
    img_list.sort()
    
    # Generate label predictions
    for img_name in img_list:
        # load image and convert to BGR format
        img_path = os.path.join(rgb_folder, img_name)
        image = cv2.imread(img_path)
        
        # Visualize the predictions
        predictions, visualized_output = demo.run_on_image(image)
        panoptic_seg, seg_infos = predictions['panoptic_seg']
        
        # create output
        panoptic_mask = np.zeros((panoptic_seg.shape[0], panoptic_seg.shape[1], 3), dtype=np.uint8)
        for sinfo in seg_infos:
            try:
                panoptic_mask[panoptic_seg.cpu().numpy() == sinfo['id']] = coco_viplanner_color_mapping[sinfo['category_id']]
            except KeyError:
                print(f"Category {sinfo['category_id']+1} not found in coco_viplanner_cls_mapping.")
                panoptic_mask[panoptic_seg.cpu().numpy() == sinfo['id']] = viplanner_sem_class_color_map['static']
        im.set_data(visualized_output.get_image()[:, :, ::-1])
        
        # save the bgr 
        cv2.imwrite(os.path.join(sem_folder, img_name), panoptic_mask)
        
        # Redraw the plot
        plt.draw()
        plt.pause(0.1)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prelabel images with a given model')
    parser.add_argument('-d', '--dataset_dir', type=str, help='Directory of the dataset',
                        default="/home/pascal/SemNav/imperative_learning/data/nomoko_zurich/rgb")
    parser.add_argument('-m', '--m2f_model', type=str, help='Path to the model',
                        default="/home/pascal/SemNav/sem_seg/m2f_model/coco/panoptic/swin/model_final_9fd0ae.pkl")
    parser.add_argument('-c', '--m2f_config', type=str, help='Path to the config',
                        default="/home/pascal/SemNav/sem_seg/m2f_model/coco/panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml")
    args = parser.parse_args()
    
    # run on directory
    m2f_run_on_folder(args.dataset_dir, args.m2f_model, args.m2f_config)

# EoF
