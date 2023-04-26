"""
M2F Overfit with own training dataset  
(Trainer modified from https://github.com/facebookresearch/Mask2Former/blob/main/train_net.py)

@author: Pascal Roth
@email:  rothpa@student.ethz.ch

@brief:  M2F overfit with own training dataset
"""

# python
import os
import copy
import itertools
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Set
import torch
import json
import random

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

# detectron2
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_panoptic, register_coco_panoptic_separated
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# viplanner
from viplanner.config import Mask2FormerCfg, SegmentsCfg
from viplanner.third_party.mask2former.mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
from viplanner.third_party.mask2former.datasets.prepare_coco_semantic_annos_from_panoptic_annos import separate_coco_semantic_from_panoptic


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

  
class M2FOverfit:
    """
    Register Dataset in the Detectron2 format
    """

    def __init__(
        self, 
        m2f_cfg: Mask2FormerCfg,
        segments_cfg: SegmentsCfg,
        ) -> None:

        self.m2f_cfg = m2f_cfg
        self.segments_cfg = segments_cfg
          
        # prepare datasets 
        self.name_coco_train    = "coco_2017_train_panoptic_own"
        self.name_coco_val      = "coco_2017_val_panoptic_own"
        self.name_zurich_train  = "zurich_panoptic_train"
        return

    def prepare_datasets(self) -> None:
        # metadata
        coco_meta = {
            "thing_classes": MetadataCatalog.data["coco_2017_train_panoptic"].thing_classes,
            "thing_colors":  MetadataCatalog.data["coco_2017_train_panoptic"].thing_colors,
            "thing_dataset_id_to_contiguous_id": MetadataCatalog.data["coco_2017_train_panoptic"].thing_dataset_id_to_contiguous_id,
            "stuff_classes": MetadataCatalog.data["coco_2017_train_panoptic"].stuff_classes,
            "stuff_colors":  MetadataCatalog.data["coco_2017_train_panoptic"].stuff_colors,
            "stuff_dataset_id_to_contiguous_id": MetadataCatalog.data["coco_2017_train_panoptic"].stuff_dataset_id_to_contiguous_id,
        }
        
        # use reduced set of COCO images
        if self.m2f_cfg.coco_nb_images is not None:
            print("Selecting a subset of COCO images ...", end=" ")
            with open(os.path.join(self.m2f_cfg.coco_data_path, "annotations/panoptic_train2017.json"), "r") as file:
                coco_train_json = json.load(file)
            
            # reduce number of images
            images_filelist = [single_image["file_name"][:-4] for single_image in coco_train_json["images"]]
            annotations_filelist = [single_annotation["file_name"][:-4] for single_annotation in coco_train_json["annotations"]]
            
            # randomly select images
            selected_images = random.sample(annotations_filelist, self.m2f_cfg.coco_nb_images)
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
            json.dump(coco_train_json_reduced, open(os.path.join(self.m2f_cfg.coco_data_path, "annotations/panoptic_train2017_reduced.json"), "w"))
            print("done")
            
            # used json file
            annotation_json = "annotations/panoptic_train2017_reduced.json"
        else:
            annotation_json = "annotations/panoptic_train2017.json"
            
        # register new coco dataset with modified paths        
        register_coco_panoptic(
            name=self.name_coco_train,
            metadata=coco_meta,
            image_root=os.path.join(self.m2f_cfg.coco_data_path, "train2017"),
            panoptic_root=os.path.join(self.m2f_cfg.coco_data_path, "annotations/panoptic_train2017"),
            panoptic_json=os.path.join(self.m2f_cfg.coco_data_path, annotation_json),
        )
        if not self.m2f_cfg.use_sem_seg:
            register_coco_panoptic(
                name=self.name_coco_val,
                metadata=coco_meta,
                image_root=os.path.join(self.m2f_cfg.coco_data_path, "val2017"),
                panoptic_root=os.path.join(self.m2f_cfg.coco_data_path, "annotations/panoptic_val2017"),
                panoptic_json=os.path.join(self.m2f_cfg.coco_data_path, "annotations/panoptic_val2017.json"),
            )
        else:
            raise NotImplementedError("Also requires instances json file")
            register_coco_panoptic_separated(
                name=self.name_coco_val,
                metadata=coco_meta,
                image_root=os.path.join(self.m2f_cfg.coco_data_path, "val2017"),
                panoptic_root=os.path.join(self.m2f_cfg.coco_data_path, "annotations/panoptic_val2017"),
                panoptic_json=os.path.join(self.m2f_cfg.coco_data_path, "annotations/panoptic_val2017.json"),
                sem_seg_root=os.path.join(self.m2f_cfg.coco_data_path, "panoptic_semseg_val2017"),       
                instances_json=None,         
            )
            separate_coco_semantic_from_panoptic(
                panoptic_json=os.path.join(self.m2f_cfg.coco_data_path, "annotations/panoptic_val2017.json"),
                panoptic_root=os.path.join(self.m2f_cfg.coco_data_path, "annotations/panoptic_val2017"),
                sem_seg_root=os.path.join(self.m2f_cfg.coco_data_path, "panoptic_semseg_val2017"),
                categories=COCO_CATEGORIES
            )
        
        # add own dataset (Metadata same as the coco dataset)
        register_coco_panoptic(
            name=self.name_zurich_train,
            metadata=coco_meta,
            image_root=os.path.join(self.segments_cfg.export_file_path, "train"),
            panoptic_root=os.path.join(self.segments_cfg.export_file_path, "annotations", "images"),
            panoptic_json=os.path.join(self.segments_cfg.export_file_path, "annotations", "panoptic_zurich.json"),
        )
        print('[INFO] Dataset registered!')

    def overfit(self) -> None:
        launch(
            self.main,
            self.m2f_cfg.num_gpus,
            num_machines=self.m2f_cfg.num_machines,
            machine_rank=self.m2f_cfg.machine_rank,
            dist_url=self.m2f_cfg.dist_url,
            args=(),
        )
        
    def setup(self):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(self.m2f_cfg.config_file)
        cfg.merge_from_list(["MODEL.WEIGHTS", self.m2f_cfg.model_file])

        # change to the new datasets
        cfg["DATASETS"]["TRAIN"] = (self.name_zurich_train, )  # (self.name_coco_train, self.name_zurich_train, )
        cfg["DATASETS"]["TEST"]  = (self.name_coco_val, )
        # change batchsize and epochs
        cfg['SOLVER']['IMS_PER_BATCH'] = self.m2f_cfg.batch_size
        iter_steps = int(self.m2f_cfg.epochs * self.m2f_cfg.coco_nb_images / self.m2f_cfg.batch_size)
        cfg['SOLVER']['MAX_ITER']      = int(iter_steps * 1.2)
        cfg['SOLVER']['STEPS']         = (int(iter_steps / 2), iter_steps)

        # change output dir
        cfg.OUTPUT_DIR = self.m2f_cfg.output_path
        
        # disable instance testing  (instance json not available)
        cfg["MODEL"]["MASK_FORMER"]["TEST"]["INSTANCE_ON"] = False
        if not self.m2f_cfg.use_sem_seg:
            cfg["MODEL"]["MASK_FORMER"]["TEST"]["SEMANTIC_ON"] = False
            
        cfg.freeze()
        default_setup(cfg, self.m2f_cfg)
        # Setup logger for "mask_former" module
        setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
        return cfg

    def main(self):
        cfg = self.setup()
        print('[INFO] Config setup done!')
        self.prepare_datasets()
        
        if self.m2f_cfg.eval_only:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=self.m2f_cfg.resume
            )
            res = Trainer.test(cfg, model)
            if cfg.TEST.AUG.ENABLED:
                res.update(Trainer.test_with_TTA(cfg, model))
            if comm.is_main_process():
                verify_results(cfg, res)
            return res

        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=self.m2f_cfg.resume)
        return trainer.train()   
    
     
if __name__ == '__main__':
    m2f_cfg = Mask2FormerCfg()
    segments_cfg = SegmentsCfg()    
    overfit = M2FOverfit(m2f_cfg=m2f_cfg, segments_cfg=segments_cfg)
    overfit.overfit()
    
# EoF
    