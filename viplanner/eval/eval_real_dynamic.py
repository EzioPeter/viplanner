"""
@author: Pascal Roth
@email:  rothpa@student.ethz.ch

@brief:  Evaluation script for real world
"""

# python
import os
import argparse
import numpy as np
from tqdm import tqdm

# viplanner
from viplanner.utils.eval_utils import BaseEvaluator


class RealWorldEvaluatorDynamic(BaseEvaluator):
    """
    VIPlanner Evaluator for real world data. Evaluated are the walked path regarding their cost, path length and success. 
    Data should be extracted from rosbags using the scripts in viplanner/scripts/rosbag_extractor.py
    """

    def __init__(
            self,
            tolerance: float,
            data_dir: str,
            debug: bool = False,
        ) -> None:
        """
        tolerance:
        data_dir: directory where the data is saved, expected structure:
            - data_dir
                - odom_goal.txt     # goal position in odom frame
                - odom_base.txt     # base position in odom frame
        debug: if true, debug information is printed
        """
        # add params
        self.data_dir = data_dir
        self.debug = debug

        # include cost map params if available
        cost_map_name = "cost_map_sem"
        cost_map_dir  = os.path.join(self.data_dir, "maps")
        if not os.path.exists(cost_map_dir):
            cost_map_dir = None
            print("[INFO]  No cost map directory found, skipping cost map evaluation.")
        
        # init base class
        super().__init__(tolerance, cost_map_dir, cost_map_name)

        # data
        self.odom_goal: np.ndarray = None
        self.odom_base: np.ndarray = None
        self.time_goal: np.ndarray = None
        self.time_base: np.ndarray = None
    
    def eval(self) -> None:
        # load data
        self._load_data()
        self.set_nbr_paths(len(self.time_goal))
        self.create_buffers()
        # separate data
        path_masks = self._sep_data()
        # evaluate
        for idx, curr_goal in enumerate(tqdm(self.odom_goal, desc="Process goal points")):
            if not any(path_masks[idx]):
                continue
            
            odom_points = self.odom_base[path_masks[idx]]

            self.length_goal[idx] = np.linalg.norm(odom_points[0] - curr_goal)
            self.length_path[idx] = sum([np.sqrt((x2-x1)**2 + (y2-y1)**2) for (x1,y1), (x2,y2) in zip(odom_points, odom_points[1:])])
            self.goal_distances[idx] = np.linalg.norm(odom_points[-1] - curr_goal)

        # plot results
        eval_dir = os.path.join(self.data_dir, "eval_dynamic")
        self.plt_single_model(eval_dir)
        return

    def _load_data(self) -> None:
        odom_goal = np.loadtxt(os.path.join(self.data_dir, "odom_goal.txt"))
        odom_base = np.loadtxt(os.path.join(self.data_dir, "odom_base.txt"))
        self.odom_goal = odom_goal[:, :2]
        self.odom_base = odom_base[:, :2]   

        # convert time
        self.time_goal = odom_goal[:, 3] + odom_goal[:, 4] / 1e9
        self.time_base = odom_base[:, 7] + odom_base[:, 8] / 1e9
        return

    def _sep_data(self) -> None:
        path_masks = np.zeros((len(self.time_goal), len(self.time_base)), dtype=bool)
        for idx, goal_time in enumerate(self.time_goal[:-1]):
            if idx == len(self.time_goal) - 1:
                path_masks[idx] = self.time_base >= goal_time
            else:
                path_masks[idx] = np.logical_and(self.time_base >= goal_time, self.time_base <= self.time_goal[idx+1])
        return path_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Model Eval', description='Evaluate the VIPlanner paths walked in the real-world')
    parser.add_argument('--data_dir',  type=str, help='Path to data directory (should contain goal and odom data)', 
                        default="/home/pascal/viplanner/env/anymal/2023_03_23_rsl")
    parser.add_argument('--tolerance', type=float, help='Tolerance to the goal to be considered reached',
                        default=0.5)
    parser.add_argument('--save_viz', action='store_true', help='Save visualizations of the predictions')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()
    
    evaluator = RealWorldEvaluatorDynamic(
        tolerance=args.tolerance,
        data_dir=args.data_dir,
        debug=args.debug,
    )
    evaluator.eval()

# EoF
