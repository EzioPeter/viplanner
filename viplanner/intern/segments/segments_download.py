"""
Script to convert the segments.ai dataset to COCO panoptic format

@author: Pascal Roth
@email:  rothpa@student.ethz.ch

@brief:  Segment to COCO panoptic format conversion
"""

import json

# python
import os
import shutil

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
from tqdm import tqdm

# viplanner
from .config import SegmentsCfg

_COCO_MAPPING_UNIQUE = {
    "road": "road",
    "sidewalk": "pavement-merged",
    "crosswalk": "pavement-merged",
    "floor": "floor-other-merged",
    "gravel": "gravel",
    "sand": "sand",
    "snow": "snow",
    "stairs": "stairs",
    "person": "person",
    "anymal": "bird",
    "vehicle": "car",
    "on_rails": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "building": "building-other-merged",
    "wall": "wall-other-merged",
    "fence": "fence-merged",
    "bridge": "bridge",
    "tunnel": "bridge",
    "pole": "parking meter",
    "traffic_sign": "stop sign",
    "traffic_light": "traffic light",
    "bench": "bench",
    "vegetation": "tree-merged",
    "terrain": "grass-merged",
    "water_surface": "river",
    "sky": "sky-other-merged",
    "background": "sky-other-merged",
    "dynamic": "backpack",
    "static": "unknown",
    "furniture": "chair",
    "door": "door-stuff",
    "ceiling": "ceiling-merged",
    "indoor_soft": "towel",
}


def main(cfg: SegmentsCfg) -> None:
    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient(cfg.api_key)
    release = client.get_release(cfg.dataset_name, cfg.version)
    dataset = SegmentsDataset(release, labelset="ground-truth", filter_by=["labeled"])
    # Export to COCO panoptic format
    export_dataset(
        dataset,
        export_format=cfg.export_format,
        export_folder=cfg.export_dir_path,
    )
    if os.path.exists(os.path.join(cfg.export_dir_path, "segments")):
        shutil.rmtree(os.path.join(cfg.export_dir_path, "segments"))
    shutil.move("./segments", os.path.join(cfg.export_dir_path))
    # check for expected structure
    json_file = os.path.join(
        cfg.export_dir_path,
        (f"export_{cfg.export_format}_{cfg.dataset_name.replace('/', '_')}_{cfg.version}.json"),
    )
    img_dir = os.path.join(
        cfg.export_dir_path,
        "segments",
        cfg.dataset_name.replace("/", "_"),
        cfg.version,
    )
    target_dir = os.path.join(
        cfg.export_dir_path,
        "segments",
        cfg.dataset_name.replace("/", "_"),
        "annotations",
    )
    assert os.path.exists(img_dir), f"Image directory {img_dir} does not exist"
    assert os.path.exists(json_file), f"JSON file {json_file} does not exist"
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)

    with open(json_file) as f:
        annotation_file = json.load(f)

    img_names = os.listdir(img_dir)
    img_list = [single_image for single_image in img_names if cfg.export_format in single_image]

    map_coco_class_to_coco_color = {}
    map_coco_class_to_coco_id = {}
    for class_item in COCO_CATEGORIES:
        map_coco_class_to_coco_color[class_item["name"]] = class_item["color"]
        map_coco_class_to_coco_id[class_item["name"]] = class_item["id"]

    map_vip_id_to_coco_color = {}
    map_vip_id_to_coco_id = {}
    for category_dict in annotation_file["categories"]:
        if category_dict["name"] == "static" or category_dict["name"] == "unknown":
            map_vip_id_to_coco_id[category_dict["id"]] = 0
            map_vip_id_to_coco_color[category_dict["id"]] = [0, 0, 0]
            unknown_class_id = category_dict["id"]
            continue
        map_vip_id_to_coco_id[category_dict["id"]] = map_coco_class_to_coco_id[
            _COCO_MAPPING_UNIQUE[category_dict["name"]]
        ]
        map_vip_id_to_coco_color[category_dict["id"]] = map_coco_class_to_coco_color[
            _COCO_MAPPING_UNIQUE[category_dict["name"]]
        ]

    # modify annotations
    annotations = []
    for idx, annotation_dict in enumerate(tqdm(annotation_file["annotations"], desc="Converting Annotations")):
        segments_new = annotation_dict.copy()
        rm_idx = []
        for idx, segment in enumerate(annotation_dict["segments_info"]):
            if (segment["category_id"]) == unknown_class_id:
                rm_idx.append(idx)
            else:
                segments_new["segments_info"][idx]["category_id"] = map_vip_id_to_coco_id[segment["category_id"]]
        segments_new["file_name"] = segments_new["file_name"].replace("_label_ground-truth_coco-panoptic", "")
        segments_new["segments_info"] = [
            single_segment for idx, single_segment in enumerate(segments_new["segments_info"]) if idx not in rm_idx
        ]
        annotations.append(segments_new)
    panoptic_file = {
        "info": annotation_file["info"],
        "categories": COCO_CATEGORIES,
        "images": annotation_file["images"],
        "annotations": annotations,
    }
    with open(os.path.join(target_dir, "panoptic_zurich.json"), "w") as file:
        json.dump(panoptic_file, file)

    # move images
    for single_img in tqdm(img_list, desc="Moving images"):
        shutil.move(
            os.path.join(img_dir, single_img),
            os.path.join(
                target_dir,
                "images",
                single_img.replace("_label_ground-truth_coco-panoptic", ""),
            ),
        )

    train_img_dir = os.path.join(
        cfg.export_dir_path,
        "segments",
        cfg.dataset_name.replace("/", "_"),
        "train",
    )
    # cleanup
    for img in os.listdir(img_dir):
        if "label" in img:
            os.remove(os.path.join(img_dir, img))
    shutil.move(img_dir, train_img_dir)

    # change images from png to jpg
    # for img in os.listdir(train_img_dir):
    #     if img.endswith(".png"):
    #         os.rename(os.path.join(train_img_dir, img), os.path.join(train_img_dir, img.replace(".png", ".jpg")))
    return


if __name__ == "__main__":
    # parse arguments
    cfg = SegmentsCfg()
    main(cfg)

# EoF
