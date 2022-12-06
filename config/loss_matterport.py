#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      Hand crafted loss for Matterport3D mp40 classes
"""

TRAVERSABLE_LOSS = 0
OBSTACLE_LOSS = 1

MATTERPORT_LOSS = {
    'void' : OBSTACLE_LOSS,
    'wall' : OBSTACLE_LOSS,
    'floor' : TRAVERSABLE_LOSS,
    'chair' : OBSTACLE_LOSS,
    'door' : OBSTACLE_LOSS,
    'table' : OBSTACLE_LOSS,
    'picture' : OBSTACLE_LOSS,
    'cabinet' : OBSTACLE_LOSS,
    'cushion' : OBSTACLE_LOSS,
    'window' : OBSTACLE_LOSS,
    'sofa' : OBSTACLE_LOSS,
    'bed' : OBSTACLE_LOSS,
    'curtain' : OBSTACLE_LOSS,
    'chest_of_drawers' : OBSTACLE_LOSS,
    'plant' : OBSTACLE_LOSS,
    'sink' : OBSTACLE_LOSS,
    'stairs' : TRAVERSABLE_LOSS,
    'ceiling' : OBSTACLE_LOSS,
    'toilet' : OBSTACLE_LOSS,
    'stool' : OBSTACLE_LOSS,
    'towel' : OBSTACLE_LOSS,
    'mirror' : OBSTACLE_LOSS,
    'tv_monitor' : OBSTACLE_LOSS,
    'shower' : OBSTACLE_LOSS,
    'column' : OBSTACLE_LOSS,
    'bathtub' : OBSTACLE_LOSS,
    'counter' : OBSTACLE_LOSS,
    'fireplace' : OBSTACLE_LOSS,
    'lighting' : OBSTACLE_LOSS,
    'beam' : OBSTACLE_LOSS,
    'railing' : OBSTACLE_LOSS,
    'shelving' : OBSTACLE_LOSS,
    'blinds' : OBSTACLE_LOSS,
    'gym_equipment' : OBSTACLE_LOSS,
    'seating' : OBSTACLE_LOSS,
    'board_panel' : OBSTACLE_LOSS,
    'furniture' : OBSTACLE_LOSS,
    'appliances' : OBSTACLE_LOSS,
    'clothes' : OBSTACLE_LOSS,
    'objects' : OBSTACLE_LOSS,
    'misc' : OBSTACLE_LOSS,
    'unlabele' : OBSTACLE_LOSS,
}
    