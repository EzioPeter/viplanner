"""
Test to improve the speed of PyTorch models
"""

# python
import torch
# from speedster import optimize_model, save_model, load_model
from PIL import Image
import os
import numpy as np
import time

# viplanner
from third_party.mask2former.mask2former import add_maskformer2_config
from detectron2.config import get_cfg, CfgNode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

def get_m2f_model(cfg_path: str, model_weights: str):
    cfg: CfgNode = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(['MODEL.WEIGHTS', model_weights])
    cfg.freeze()
    model = build_model(cfg)
    
    checkpointer = DetectionCheckpointer(model)
    print("Model weights loaded from: ", cfg.MODEL.WEIGHTS)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    return model, cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD

def load_images_from_directory(directory_path, batch_size=1):
    image_list = []
    counter = 0
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image = Image.open(os.path.join(directory_path, filename))
            image = np.array(image)
            image_list.append(image)
            counter += 1
        if counter == batch_size:
            break
    return image_list

# get m2f model
image_path = "/home/pascal/SemNav/env/anymal/2023_01_26_eth/2023-01-26-16-10-24_rgb_front_rear_lidar_depth_left_right_mission_0"
m2f_cfg_path = "/home/pascal/SemNav/imperative_learning/models/m2f_model/coco/panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml"
m2f_weights = "/home/pascal/SemNav/imperative_learning/models/m2f_model/coco/panoptic/swin/model_final_9fd0ae.pkl"
batch_size = 1

images = load_images_from_directory(image_path, batch_size=batch_size)
model, pixel_mean, pixel_std = get_m2f_model(m2f_cfg_path, m2f_weights)
model.eval()
images = np.stack(images)
images = images.transpose(0, 3, 1, 2)
inputs = [{"image": torch.from_numpy(image_single), "height": images.shape[2], "width": images.shape[3]} for image_single in images]

start = time.time()
output = model(inputs)[0]
normal_time = time.time() - start
print("Normal model inference time: ", normal_time)

"""Torch compile
https://pytorch.org/get-started/pytorch-2.0/#user-experience
"""

model_compiled = torch.compile(model)
# compile model
output = model_compiled(inputs)[0]

# run faster
start = time.time()
output = model_compiled(inputs)[0]
normal_time = time.time() - start
print("Compiled model inference time: ", normal_time)


"""Speedster
https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/speedster

Install:
pip install speedster
python -m nebullvm.installers.auto_installer --compilers all
"""
          
# #2 Run Speedster optimization
# optimized_model = optimize_model(
#     model, 
#     input_data=inputs, 
#     optimization_time="constrained",
#     metric_drop_ths=0.05
# )

# #3 Save the optimized model
# save_model(optimized_model, "model_save_path")

# #4 Load and run your PyTorch accelerated model in production
# from speedster import load_model

# optimized_model = load_model("model_save_path")

# start = time.time()
# output = optimized_model(inputs)[0]
# optimized_time = time.time() - start
# print("Optimized model inference time: ", optimized_time, " (", optimized_time/normal_time, "x speedup)")

