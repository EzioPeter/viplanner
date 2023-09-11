"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      warp utils
"""
import argparse

# python
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# m2f
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

# viplanner
from viplanner.config import Mask2FormerCfg, VIPlannerSemMetaHandler, get_class_for_id
from viplanner.third_party.mask2former.mask2former import add_maskformer2_config


class M2FWrapper:
    def __init__(self, m2f_cfg: Mask2FormerCfg):
        self.m2f_cfg = m2f_cfg

        # load m2f model
        self.predictor: DefaultPredictor = None
        self.load_m2f_pred()

        # load mapping to viplanner semantic classes
        self.viplanner_meta = VIPlannerSemMetaHandler()
        self.coco_viplanner_cls_mapping = get_class_for_id()
        self.viplanner_sem_class_color_map = self.viplanner_meta.class_color
        self.coco_viplanner_color_mapping = {}
        for (
            coco_id,
            viplanner_cls_name,
        ) in self.coco_viplanner_cls_mapping.items():
            self.coco_viplanner_color_mapping[coco_id] = (
                self.viplanner_meta.class_color[viplanner_cls_name]
            )

        return

    def load_m2f_pred(self) -> None:
        setup_logger(name="fvcore")
        logger = setup_logger()

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(self.m2f_cfg.config_file)
        cfg.merge_from_list(["MODEL.WEIGHTS", self.m2f_cfg.model_file])
        cfg.freeze()

        print("Network expects images of type: ", cfg.INPUT.FORMAT)

        # create predictor
        self.predictor = DefaultPredictor(cfg)

        return

    def run_image(self, img: np.ndarray) -> np.ndarray:
        """
        Run m2f on image (BGR format) and return semantic image with viplanner classes (RGB format)
        :param img: image in BGR format
        :return: semantic image in RGB format
        """
        # make the prediction
        predictions = self.predictor(img)
        return self._create_mask(predictions)

    def run_on_folder(
        self,
        data_folder: str,
        show_pred: bool = True,
        run_on_existing_files: bool = True,
        sem_folder_name: str = "semantics",
    ) -> None:
        """
        Run m2f on folder
        :param data_folder: folder with rgb images

        :return: None
        """
        # check folder and create semantics folder
        assert os.path.isdir(
            data_folder
        ), f"Folder {data_folder} does not exist!"
        parent_folder, _ = os.path.split(data_folder)
        sem_folder = os.path.join(parent_folder, sem_folder_name)
        os.makedirs(sem_folder, exist_ok=True)

        img_list = os.listdir(data_folder)
        if not run_on_existing_files:
            # check if and which files are already in the directory, only progress for the non-included files
            files_in_directory_set = set(os.listdir(sem_folder))
            filenames_set = set(img_list)
            img_list = list(filenames_set - files_in_directory_set)
            if len(img_list) == 0:
                return
        else:
            # load all images from the folder
            img_list.sort()

        if show_pred:
            # init plot to show predictions
            fig, ax = plt.subplots()
            im = ax.imshow(np.zeros((1080, 1440, 3), dtype=np.uint8))

        # Generate label predictions
        for img_name in tqdm(img_list, desc="Semantic Predicted"):
            # load image and convert to BGR format
            img_path = os.path.join(data_folder, img_name)
            image = cv2.imread(img_path)

            # make the prediction
            panoptic_mask = self.run_image(image)

            # save image
            sem_img_path = os.path.join(sem_folder, img_name)
            cv2.imwrite(
                sem_img_path, cv2.cvtColor(panoptic_mask, cv2.COLOR_RGB2BGR)
            )

            # show image
            if show_pred:
                im.set_data(panoptic_mask)
                plt.pause(0.001)

        return

    def _create_mask(self, predictions: dict) -> np.ndarray:
        panoptic_seg, seg_infos = predictions["panoptic_seg"]

        # create output
        panoptic_mask = np.zeros(
            (panoptic_seg.shape[0], panoptic_seg.shape[1], 3), dtype=np.uint8
        )
        for sinfo in seg_infos:
            try:
                panoptic_mask[panoptic_seg.cpu().numpy() == sinfo["id"]] = (
                    self.coco_viplanner_color_mapping[sinfo["category_id"]]
                )
            except KeyError:
                print(
                    f"Category {sinfo['category_id']+1} not found in"
                    " coco_viplanner_cls_mapping."
                )
                panoptic_mask[panoptic_seg.cpu().numpy() == sinfo["id"]] = (
                    self.viplanner_sem_class_color_map["static"]
                )

        return panoptic_mask


if __name__ == "__main__":
    m2f_cfg = Mask2FormerCfg()
    wrapper = M2FWrapper(m2f_cfg)

    parser = argparse.ArgumentParser(
        description="Prelabel images with a given model"
    )
    parser.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        help="Directory of the dataset",
        default="/media/pascal/NavigationData/PascalRothData/bag/2023_09_07_stairs_both_door/bgr",
    )
    parser.add_argument(
        "-s",
        "--sem_dir_name",
        type=str,
        help="Name of the new directory for the semantic images",
        default="semantics",
    )
    args = parser.parse_args()

    # run on directory
    wrapper.run_on_folder(args.dataset_dir, sem_folder_name=args.sem_dir_name)

# EoF
