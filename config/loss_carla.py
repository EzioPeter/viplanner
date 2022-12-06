#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      Hand crafted loss for Cityscapes Classes (used for the semantics of the CARLA dataset)
"""

OBSTACLE_LOSS = 1
TRAVERSABLE_LOSS = 0
#TODO: make difference between road and sidewalk and vegetation and terrain

CARLA_LOSS = {
    # flat
    "road" : TRAVERSABLE_LOSS,
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
    "terrain" : TRAVERSABLE_LOSS,
    # sky
    "sky" : OBSTACLE_LOSS,
    # void
    "ground" : OBSTACLE_LOSS,
    "dynamic" : OBSTACLE_LOSS,
    "static" : OBSTACLE_LOSS,
}

# EoF
