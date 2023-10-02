"""
Script to prelabel real-world RGB images which label can be fine-tuned in segments.ai

@author: Pascal Roth
@email:  rothpa@student.ethz.ch

@brief:  Prelabel images with a given model
"""

import matplotlib.pyplot as plt

# python
import numpy as np

# segments.ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import bitmap2file

from viplanner.config import Mask2FormerCfg, VIPlannerSemMetaHandler, get_class_for_id

# viplanner
from viplanner.intern.m2f.m2f_utils import M2FWrapper
from viplanner.intern.segments.config import SegmentsCfg


def main(segments_cfg: SegmentsCfg, m2f_cfg: Mask2FormerCfg):
    # create client
    client = SegmentsClient(segments_cfg.api_key)
    release = client.get_release(segments_cfg.dataset_name, segments_cfg.version)

    # Initialize a new dataset, this time containing only unlabeled images
    dataset = SegmentsDataset(release, labelset="ground-truth", filter_by="UNLABELED")

    # get m2f model
    m2f_wrapper = M2FWrapper(m2f_cfg)

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
        image = np.asarray(sample["image"])
        image = image[:, :, ::-1]

        # Visualize the predictions
        predictions = m2f_wrapper.predictor(image)
        segmentation_bitmap, seg_infos = predictions["panoptic_seg"]

        # create output with VIPLanner Semantic Classes
        for sinfo in seg_infos:
            sinfo["category_id"] = (
                coco_viplanner_mapping[sinfo["category_id"]] + 1
            )  # +1 because segments ids start at 1

        # Upload the predictions to Segments.ai
        file = bitmap2file(segmentation_bitmap.cpu().numpy().astype(np.uint32))
        asset = client.upload_asset(file, "label.png")
        attributes = {
            "format_version": "0.1",
            "annotations": seg_infos,
            "segmentation_bitmap": {"url": asset.url},
        }
        client.add_label(
            sample["uuid"],
            "ground-truth",
            attributes,
            label_status="PRELABELED",
        )

        im.set_data(m2f_wrapper._create_mask(predictions=predictions))
        # Redraw the plot
        plt.draw()
        plt.pause(0.1)
    return


if __name__ == "__main__":
    segments_cfg = SegmentsCfg()
    m2f_cfg = Mask2FormerCfg()
    main(segments_cfg, m2f_cfg)

# EoF
