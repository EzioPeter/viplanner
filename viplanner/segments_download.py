"""
Script to convert the segments.ai dataset to COCO panoptic format

@author: Pascal Roth
@email:  rothpa@student.ethz.ch

@brief:  Segment to COCO panoptic format conversion
"""

# python
import os
import json
import shutil
from tqdm import tqdm
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

# viplanner
from viplanner.config import VIPlannerSemMetaHandler, _COCO_MAPPING_UNIQUE, SegmentsCfg


def main(cfg: SegmentsCfg) -> None:
    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient(cfg.api_key)
    release = client.get_release(cfg.dataset_name, cfg.version)
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled'])
    # Export to COCO panoptic format
    export_dataset(dataset, export_format=cfg.export_format, export_folder=cfg.export_dir_path)
    if os.path.exists(os.path.join(cfg.export_dir_path, "segments")):
        shutil.rmtree(os.path.join(os.path.join(cfg.export_dir_path), "segments"))
    shutil.move("./segments", os.path.join(cfg.export_dir_path))
    # check for expected structure
    json_file   = os.path.join(cfg.export_dir_path, f"export_{cfg.export_format}_{cfg.dataset_name.replace('/', '_')}_{cfg.version}.json")
    img_dir     = os.path.join(cfg.export_dir_path, "segments", cfg.dataset_name.replace("/", "_"), cfg.version)
    target_dir  = os.path.join(cfg.export_dir_path, "segments", cfg.dataset_name.replace("/", "_"), "annotations")
    assert os.path.exists(img_dir), f"Image directory {img_dir} does not exist"
    assert os.path.exists(json_file), f"JSON file {json_file} does not exist"
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
        
    with open(json_file, "r") as f:
        annotation_file = json.load(f)
    
    img_names = os.listdir(img_dir)
    img_list = [single_image for single_image in img_names if cfg.export_format in single_image]
    
    # map: segments_ai color -> coco color and segments_ai id --> coco id
    sem_handler = VIPlannerSemMetaHandler()
    
    map_coco_class_to_coco_color = {}
    map_coco_class_to_coco_id = {}
    for class_item in COCO_CATEGORIES:
        map_coco_class_to_coco_color[class_item['name']] = class_item['color']
        map_coco_class_to_coco_id[class_item['name']] = class_item['id']
        
    map_vip_id_to_coco_color = {}
    map_vip_id_to_coco_id = {}
    for class_name, class_id in sem_handler.class_id.items():
        if class_name == "static":
            unknown_class_id = class_id
            continue
        map_vip_id_to_coco_id[class_id] = map_coco_class_to_coco_id[_COCO_MAPPING_UNIQUE[class_name]]
        map_vip_id_to_coco_color[class_id] = map_coco_class_to_coco_color[_COCO_MAPPING_UNIQUE[class_name]]

    # modify annotations
    annotations = []
    for idx, annotation_dict in enumerate(tqdm(annotation_file["annotations"], desc="Converting Annotations")):
        segments_new = annotation_dict
        rm_idx = []
        for idx, segment in enumerate(annotation_dict['segments_info']):
            if (segment["category_id"] - 1) == unknown_class_id:
                rm_idx.append(idx)
            else:
                segments_new['segments_info'][idx]["category_id"] = map_vip_id_to_coco_id[segment["category_id"] -1]
        segments_new["file_name"] = segments_new["file_name"].replace('_label_ground-truth_coco-panoptic', "")
        segments_new['segments_info'] = [single_segment for idx, single_segment in enumerate(segments_new['segments_info']) if idx not in rm_idx]
        annotations.append(segments_new)
    panoptic_file = {'info': annotation_file['info'], 'categories': COCO_CATEGORIES, 'images': annotation_file["images"], 'annotations': annotations}
    json.dump(panoptic_file, open(f"{target_dir}/panoptic_zurich.json", "w"))
    
    # move images
    for single_img in tqdm(img_list, desc="Moving images"):
        shutil.move(os.path.join(img_dir, single_img), os.path.join(target_dir, "images", single_img.replace('_label_ground-truth_coco-panoptic', "")))

    # cleanup 
    for img in os.listdir(img_dir):
        if "label" in img:
            os.remove(os.path.join(img_dir, img))
    shutil.move(img_dir, os.path.join(cfg.export_dir_path, "segments", cfg.dataset_name.replace("/", "_"), "train"))
    
    return


if __name__ == '__main__':
    # parse arguments
    cfg = SegmentsCfg()
    main(cfg)
    
# EoF
