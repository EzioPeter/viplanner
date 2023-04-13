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

# viplanner
from utils.m2f_utils import load_m2f_demo
from config import VIPlannerSemMetaHandler, get_class_for_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prelabel images with a given model')
    parser.add_argument('-d', '--dataset_name', type=str, help='Name of the dataset',
                        default="leggedrobotics/urban_navigation")
    parser.add_argument('-m', '--m2f_model', type=str, help='Path to the model',
                        default="/home/pascal/SemNav/sem_seg/m2f_model/coco/panoptic/swin/model_final_9fd0ae.pkl")
    parser.add_argument('-c', '--m2f_config', type=str, help='Path to the config',
                        default="/home/pascal/SemNav/sem_seg/m2f_model/coco/panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml")
    parser.add_argument('-v', '--version', type=str, default="v0.2", help='Release Version of Mask2Former')
    return parser.parse_args()


def main(args: argparse.Namespace):
    # create client
    client = SegmentsClient("ee0a626ee7c160e6c841dcd59743b811bf25c774")
    release = client.get_release(args.dataset_name, args.version)
    
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
