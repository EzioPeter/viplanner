"""
Script to prelabel real-world RGB images which label can be fine-tuned in segments.ai

@author: Pascal Roth
@email:  rothpa@student.ethz.ch

@brief:  Prelabel images with a given model
"""

# python
import argparse
import numpy as np
import matplotlib.pyplot as plt

# segments.ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import bitmap2file

# m2f
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from third_party.mask2former.mask2former import add_maskformer2_config
from third_party.mask2former.demo.predictor import VisualizationDemo

# viplanner
from config import VIPlannerSemMetaHandler, get_class_for_id

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prelabel images with a given model')
    parser.add_argument('-d', '--dataset_name', type=str, help='Name of the dataset',
                        default="leggedrobotics/urban_navigation")
    parser.add_argument('-m', '--m2f_model', type=str, help='Path to the model',
                        default="/home/pascal/SemNav/sem_seg/m2f_model/coco/panoptic/swin/model_final_9fd0ae.pkl")
    parser.add_argument('-c', '--m2f_config', type=str, help='Path to the config',
                        default="/home/pascal/SemNav/sem_seg/m2f_model/coco/panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml")
    return parser.parse_args()


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


def main(args: argparse.Namespace):
    # create client
    client = SegmentsClient("ee0a626ee7c160e6c841dcd59743b811bf25c774")
    release = client.get_release(args.dataset_name, "v0.1")
    
    # Initialize a new dataset, this time containing only unlabeled images
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by='UNLABELED')
    
    # get m2f model
    demo = load_m2f_demo(args.m2f_model, args.m2f_config)
    
    # get mapping from coco to viplanner semantic classes
    viplanner_meta = VIPlannerSemMetaHandler()
    coco_viplanner_cls_mapping = get_class_for_id()
    coco_viplanner_mapping = {}
    for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
        coco_viplanner_mapping[coco_id] = viplanner_meta.class_id[viplanner_cls_name]
    
    # init plot to show predictions
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((1080, 1440, 3), dtype=np.uint8))

    # Generate label predictions
    for sample in dataset:
        # load image and convert to BGR format
        image = np.asarray(sample['image'])
        image = image[:, :, ::-1]
        
        # Visualize the predictions
        predictions, visualized_output = demo.run_on_image(image)
        segmentation_bitmap, seg_infos = predictions['panoptic_seg']
        
        # create output with VIPLanner Semantic Classes 
        for sinfo in seg_infos:
            sinfo['category_id'] = coco_viplanner_mapping[sinfo['category_id']] + 1  # +1 because segments ids start at 1
                   
        # Upload the predictions to Segments.ai
        file = bitmap2file(segmentation_bitmap.cpu().numpy().astype(np.uint32))
        asset = client.upload_asset(file, 'label.png')    
        attributes = {
            'format_version': '0.1',
            'annotations': seg_infos,
            'segmentation_bitmap': { 'url': asset.url },
        }
        client.add_label(sample['uuid'], 'ground-truth', attributes, label_status='PRELABELED')

        im.set_data(visualized_output.get_image()[:, :, ::-1])
        # Redraw the plot
        plt.draw()
        plt.pause(0.1)
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
# EoF
