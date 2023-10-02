import json
import os

import numpy as np

coco_data_path = "/home/pascal/viplanner/imperative_learning/data/coco"
intended_ids = {
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    10,
    11,
    13,
    14,
    15,
    64,
    95,
    112,
    118,
    128,
    149,
    161,
    171,
    175,
    176,
    177,
    178,
    184,
    185,
    187,
    190,
    191,
    193,
    194,
    197,
    198,
    199,
}

print("Selecting a subset of COCO images ...")


def change_annotation(annotation_file: str, matching: int = 9):
    with open(os.path.join(coco_data_path, annotation_file)) as file:
        coco_train_json = json.load(file)

    # reduce number of images
    images_filelist = [single_image["file_name"][:-4] for single_image in coco_train_json["images"]]
    annotations_filelist = [single_annotation["file_name"][:-4] for single_annotation in coco_train_json["annotations"]]
    annotations_ids = [
        [curr_ann["category_id"] for curr_ann in single_annotation["segments_info"]]
        for single_annotation in coco_train_json["annotations"]
    ]

    annotation_nb_intended_ids = [
        len(intended_ids.intersection(set(single_annotation_ids))) for single_annotation_ids in annotations_ids
    ]

    selected_images = (np.array(annotations_filelist)[np.array(annotation_nb_intended_ids) > matching]).tolist()
    print(f"Selected {len(selected_images)} images out of" f" {len(annotations_filelist)} images.")
    images_selected_idx = []
    annotations_selected_idx = []
    for image in selected_images:
        annotations_selected_idx.append(annotations_filelist.index(image))
        images_selected_idx.append(images_filelist.index(image))

    # save reduced json
    coco_train_json_reduced = {
        "info": coco_train_json["info"],
        "images": [coco_train_json["images"][idx] for idx in images_selected_idx],
        "annotations": [coco_train_json["annotations"][idx] for idx in annotations_selected_idx],
        "categories": coco_train_json["categories"],
    }
    path, ending = os.path.splitext(annotation_file)
    with open(os.path.join(coco_data_path, f"{path}_reduced{ending}"), "w") as file:
        json.dump(coco_train_json_reduced, file)
    print("done")


if __name__ == "__main__":
    change_annotation("annotations/panoptic_train2017.json", 10)
    change_annotation("annotations/panoptic_val2017.json", 6)
