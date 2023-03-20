#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      VIPlanner Semantic Meta Data
"""

OBSTACLE_LOSS = 1
TRAVERSABLE_LOSS = 0
ROAD_LOSS = 0.6
TERRAIN_LOSS = 0.3

# original coco meta
VIPLANNER_SEM_META = [
    # ground
    {
        'name': 'road',
        'loss': ROAD_LOSS,
        'color': [204, 204, 0],
        'ground': True,
    }, 
    {
        'name': 'sidewalk',
        'loss': TRAVERSABLE_LOSS,
        'color': [0, 255, 0],
        'ground': True,
    },
    {
        'name': 'crosswalk',
        'loss': TRAVERSABLE_LOSS,
        'color': [0, 204, 0],
        'ground': True,
    },
    {
        'name': 'floor',
        'loss': TRAVERSABLE_LOSS,
        'color': [0, 102, 0],
        'ground': True,
    },
    {
        'name': 'gravel',
        'loss': TRAVERSABLE_LOSS,
        'color': [0, 51, 0],
        'ground': True,
    },
    {
        'name': 'sand',
        'loss': TRAVERSABLE_LOSS,
        'color': [153, 255, 153],
        'ground': True,
    },
    {
        'name': 'snow',
        'loss': TRAVERSABLE_LOSS,
        'color': [107, 142, 35],
        'ground': True,
    },
    {
        'name': 'stairs',
        'loss': TRAVERSABLE_LOSS,
        'color': [0, 153, 0],
        'ground': True,
    },
    # human
    {
        'name': 'person',
        'color': [255, 0, 0],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'anymal',
        'color': [102, 0, 0],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    # vehicle
    {
        'name': 'vehicle',
        'color': [153, 76, 0],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'on_rails',
        'color': [51, 25, 0],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'motorcycle',
        'color': [204, 102, 0],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'bicycle',
        'color': [102, 51, 0],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    # construction
    {
        'name': 'building',
        'loss': OBSTACLE_LOSS,
        'color': [127, 0, 255],
        'ground': False,
    },
    {
        'name': 'wall',
        'color': [102, 0, 204],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'fence',
        'color': [76, 0, 153],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'bridge',
        'color': [51, 0, 102],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'tunnel',
        'color': [51, 0, 102],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    # object
    {
        'name': 'pole',
        'color': [0, 0, 255],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'traffic_sign',
        'color': [0, 0, 153],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'traffic_light',
        'color': [0, 0, 204],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'bench',
        'color': [0, 0, 102],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    # nature
    {
        'name': 'vegetation',
        'color': [153, 0, 153],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'terrain',
        'color': [51, 255, 51],
        'loss': TERRAIN_LOSS,
        'ground': True,
    },
    {
        'name': 'water_surface',
        'color': [204, 0, 204],
        'loss': OBSTACLE_LOSS,
        'ground': True,
    },
    # sky
    {
        'name': 'sky',
        'color': [102, 0, 51],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'background',
        'color': [102, 0, 51],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    # void outdoor
    {
        'name': 'dynamic',
        'color': [32, 32, 32],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'static',  # also everything unknown
        'color': [0, 0, 0],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    # indoor
    {
        'name': 'furniture',
        'color': [0, 0, 51],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'door',
        'color': [153, 153, 0],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'ceiling',
        'color': [25, 0, 51],
        'loss': OBSTACLE_LOSS,
        'ground': False,
    },
    {
        'name': 'indoor_soft',  # human made thing, can be walked on
        'color': [178, 255, 102],
        'loss': ROAD_LOSS,
        'ground': False,
    }
]

class VIPlannerSemMetaHandler:
    """Useful functions for handling VIPlanner semantic meta data.
    """
    
    def __init__(self) -> None:
        # meta config
        self.meta = VIPLANNER_SEM_META
        
        # class loss dict
        self.class_loss: dict = self._get_class_loss_dict()
        self.class_color: dict = self._get_class_color_dict()
        self.class_ground: dict = self._get_class_ground_dict()
        return
    
    def get_colors_for_names(self, name_list: list) -> list:
        """Get list of colors for a list of names."""
        colors = []
        name_to_color = {nc['name']: nc['color'] for nc in self.meta}
        for name in name_list:
            if name in name_to_color:
                colors.append(name_to_color[name])
        return colors

    def _get_class_loss_dict(self) -> dict:
        """Get class loss dict."""
        return {nc['name']: nc['loss'] for nc in self.meta}
    
    def _get_class_color_dict(self) -> dict:
        """Get class color dict."""
        return {nc['name']: nc['color'] for nc in self.meta}

    def _get_class_ground_dict(self) -> dict:
        """Get class ground dict."""
        return {nc['name']: nc['ground'] for nc in self.meta}
    
    @property
    def colors(self) -> list:
        """Get list of colors."""
        return list(self.class_color.values())
    
    @property
    def losses(self) -> list:
        """Get list of losses."""
        return list(self.class_loss.values())
    
    @property
    def names(self) -> list:
        """Get list of names."""
        return list(self.class_loss.keys())

    @property
    def ground(self) -> list:
        """Get list of ground."""
        return list(self.class_ground.values())


"""CLASS COLOR VISUALIZATION"""

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # init meta handler
    meta_handler = VIPlannerSemMetaHandler()
    
    # class ordering array
    cls_order = [
        ['sky', 'background', 'dynamic', 'static'],
        ['building', 'wall', 'fence', 'bridge', 'ceiling', 'vegetation', 'water_surface'],
        ['pole', 'traffic_light', 'traffic_sign', 'bench', 'furniture'],
        ['terrain', 'indoor_soft', 'sand', 'snow'],
        ['sidewalk', 'crosswalk', 'stairs', 'floor', 'gravel'],
        ['road', 'door'],
        ['motorcycle', 'vehicle', 'bicycle', 'on_rails'],
        ['anymal', 'person']
    ]
    
    # Create the 8x8 grid of subplots
    fig, axs = plt.subplots(nrows=8, ncols=7, figsize=(12, 12))

    # Loop over each subplot and plot the data
    for i in range(8):
        for j in range(7):
            ax = axs[i][j] 
            
            # Remove the axis, axis ticks, border, ...
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])            
            
            # plot color
            if j >= len(cls_order[i]):
                continue       
            ax.imshow([[tuple(meta_handler.class_color[cls_order[i][j]])]])
            ax.set_title(cls_order[i][j], fontsize=10)
            ax.set_xlabel(meta_handler.class_color[cls_order[i][j]], fontsize=8)
            
    # Set the overall title of the plot
    fig.suptitle('VIPlanner Semantic Classes Color Scheme', fontsize=16)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.tight_layout()
    
    # Show the plot
    plt.show()

# EoF
