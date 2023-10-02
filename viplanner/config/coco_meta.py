# Modified from https://github.com/google-research/deeplab2/blob/main/data/coco_constants.py
# File containing the meta info of all classes from the COCO dataset.

try:
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
except ModuleNotFoundError:
    COCO_CATEGORIES = None

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

_COCO_MAPPING = {
    "road": ["road"],
    "sidewalk": [
        "pavement-merged",
    ],
    "floor": [
        "floor-other-merged",
        "floor-wood",
        "platform",
        "playingfield",
        "rug-merged",
    ],
    "gravel": [
        "gravel",
    ],
    "stairs": [
        "stairs",
    ],
    "sand": [
        "sand",
    ],
    "snow": [
        "snow",
    ],
    "person": ["person"],
    "anymal": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "vehicle": [
        "car",
        "bus",
        "truck",
        "boat",
    ],
    "on_rails": [
        "train",
        "railroad",
    ],
    "motorcycle": [
        "motorcycle",
    ],
    "bicycle": [
        "bicycle",
    ],
    "building": [
        "building-other-merged",
        "house",
        "roof",
    ],
    "wall": [
        "wall-other-merged",
        "curtain",
        "mirror-stuff",
        "wall-brick",
        "wall-stone",
        "wall-tile",
        "wall-wood",
        "window-blind",
        "window-other",
    ],
    "fence": [
        "fence-merged",
    ],
    "bridge": [
        "bridge",
    ],
    "pole": [
        "fire hydrant",
        "parking meter",
    ],
    "traffic_sign": [
        "stop sign",
    ],
    "traffic_light": [
        "traffic light",
    ],
    "bench": [
        "bench",
    ],
    "vegetation": [
        "potted plant",
        "flower",
        "tree-merged",
        "mountain-merged",
        "rock-merged",
    ],
    "terrain": [
        "grass-merged",
        "dirt-merged",
    ],
    "water_surface": [
        "river",
        "sea",
        "water-other",
    ],
    "sky": [
        "sky-other-merged",
        "airplane",
    ],
    "dynamic": [
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "book",
        # sports
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        # kitchen
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        # food
        "banana",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "fruit",
        "food-other-merged",
        "apple",
        # computer hardware
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "laptop",
        # other
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "net",
        "paper-merged",
    ],
    "static": [
        "banner",
        "cardboard",
        "light",
        "tent",
        "unknown",
    ],
    "furniture": [
        "chair",
        "couch",
        "bed",
        "dining table",
        "toilet",
        "clock",
        "vase",
        "blanket",
        "pillow",
        "shelf",
        "cabinet",
        "table-merged",
        "counter",
        "tv",
    ],
    "door": [
        "door-stuff",
    ],
    "ceiling": ["ceiling-merged"],
    "indoor_soft": [
        "towel",
    ],
}


def get_class_for_id():
    id_to_class = {}
    for idx, id_dict in enumerate(COCO_CATEGORIES):
        success = False
        for class_name, keywords in _COCO_MAPPING.items():
            if any(keyword in id_dict["name"] for keyword in keywords):
                id_to_class[idx] = class_name
                success = True
                break
        if not success:
            print("No mapping found for {}".format(id_dict["name"]))
    return id_to_class


def get_class_for_id_mmdet(class_list: list):
    id_to_class = {}
    for idx, coco_class_name in enumerate(class_list):
        success = False
        for class_name, keywords in _COCO_MAPPING.items():
            if any(keyword in coco_class_name for keyword in keywords):
                id_to_class[idx] = class_name
                success = True
                break
        if not success:
            print("No mapping found for {}".format(coco_class_name["name"]))
    return id_to_class


if __name__ == "__main__":
    print(get_class_for_id())
