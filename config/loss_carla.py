#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      Hand crafted loss for Cityscapes Classes (used for the semantics of the CARLA dataset)
"""

# has to be the same as used in CARLA exploration
OBSTACLE_LOSS = 1
TRAVERSABLE_LOSS = 0
ROAD_LOSS = 0.6
TERRAIN_LOSS = 0.3
VOID_LOSS = 1.0

CARLA_LOSS = {
    # flat
    "road" : ROAD_LOSS,
    "sidewalk" : TRAVERSABLE_LOSS,
    "parking" : TRAVERSABLE_LOSS,
    "rail_track" : TRAVERSABLE_LOSS,
    # human
    "person" : OBSTACLE_LOSS,
    "rider" : OBSTACLE_LOSS,
    # vehicle
    "car" : OBSTACLE_LOSS,
    "truck" : OBSTACLE_LOSS,
    "bus" : OBSTACLE_LOSS,
    "on rails" : OBSTACLE_LOSS,
    "motorcycle" : OBSTACLE_LOSS,
    "bicycle" : OBSTACLE_LOSS,
    "caravan" : OBSTACLE_LOSS,
    "trailer" : OBSTACLE_LOSS,
    # construction
    "building" : OBSTACLE_LOSS,
    "wall" : OBSTACLE_LOSS,
    "fence" : OBSTACLE_LOSS,
    "guard_rail" : OBSTACLE_LOSS,
    "bridge" : OBSTACLE_LOSS,
    "tunnel" : OBSTACLE_LOSS,
    # object
    "pole" : OBSTACLE_LOSS,
    "pole_group" : OBSTACLE_LOSS,
    "traffic_sign" : OBSTACLE_LOSS,
    "traffic_light" : OBSTACLE_LOSS,
    # nature
    "vegetation" : OBSTACLE_LOSS,
    "terrain" : TERRAIN_LOSS,
    # sky
    "sky" : OBSTACLE_LOSS,
    # void
    "ground" : VOID_LOSS,
    "dynamic" : VOID_LOSS,
    "static" : VOID_LOSS,
}

# colors equal to https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
CARLA_COLOR_MAPPING = {
    # flat
    "road" :            (128, 64,128),
    "sidewalk" :        (244, 35,232),
    "parking" :         (250,170,160),
    "rail_track" :      (230,150,140),
    # human
    "person" :          (220, 20, 60),
    "rider" :           (255,  0,  0),
    # vehicle
    "car" :             (  0,  0,142),
    "truck" :           (  0,  0, 70),
    "bus" :             (  0, 60,100),
    "on rails" :        (  0, 80,100),
    "motorcycle" :      (  0,  0,230),
    "bicycle" :         (119, 11, 32),
    "caravan" :         (  0,  0, 90),
    "trailer" :         (  0,  0,110),
    # construction
    "building" :        ( 70, 70, 70),
    "wall" :            (102,102,156),
    "fence" :           (190,153,153),
    "guard_rail" :      (180,165,180),
    "bridge" :          (150,100,100),
    "tunnel" :          (150,120, 90),
    # object
    "pole" :            (153,153,153),
    "pole_group" :      (153,153,153),
    "traffic_sign" :    (220,220,  0),
    "traffic_light" :   (250,170, 30),
    # nature
    "vegetation" :      (107,142, 35),
    "terrain" :         (152,251,152),
    # sky
    "sky" :             ( 70,130,180),
    # void
    "ground" :          ( 81,  0, 81),
    "dynamic" :         (111, 74,  0),
    "static" :          (  0,  0,  0),
    "UNLABELLED" :      (  0,  0,  0),  # extra, not in cityscapes
}  

# EoF
