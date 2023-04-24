#!/usr/bin python3

"""
@author     Pascal Roth
@email      roth.pascal@outlook.de

@brief      Evaluator Base Class
"""

# python
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import yaml

# viplanner
try: 
    from config.learning_cfg import Loader as TrainCfgLoader
except ModuleNotFoundError:  # compatability with VIPlanner within isaac sim
    from omni.isaac.anymal.viplanner.src.config.learning_cfg import Loader as TrainCfgLoader


class BaseEvaluator:
    def __init__(
        self,
        tolerance: float,
    ) -> None:
        
        # args
        self.tolerance = tolerance
        
        # parameters
        self._nbr_paths: int = 0
        return

    ##
    # Properties
    ##
    
    @property
    def nbr_paths(self) -> int:
        return self._nbr_paths
    
    def set_nbr_paths(self, nbr_paths: int) -> None:
        self._nbr_paths = nbr_paths
        return
    
    ##
    # Buffer
    ##
    
    def create_buffers(self) -> None:
        self.length_goal: np.ndarray    = np.zeros((self._nbr_paths))
        self.length_path: np.ndarray    = np.zeros((self._nbr_paths))
        self.goal_distances: np.ndarray = np.zeros((self._nbr_paths))
    
    
    ##
    # Reset
    ##
    
    def reset(self) -> None:
        self.create_buffers()
        self.eval_stats = {}
        return
    
    ##
    # Eval Statistics
    ##
    
    def eval_statistics(self) -> None:
        # Evaluate results
        goal_reached = self.goal_distances < self.tolerance
        success_rate = sum(goal_reached) / len(goal_reached)  
        avg_distance_to_goal = sum(self.goal_distances) / len(self.goal_distances)
        avg_distance_to_goal_success = sum(self.goal_distances[goal_reached]) / sum(goal_reached)

        print(
            f"All path segments been passed. Results: \n"
            f"Success rate:                 {success_rate} \n"
            f"Avg goal-distance (all):      {avg_distance_to_goal} \n"
            f"Avg goal-distance (success):  {avg_distance_to_goal_success}"
        )
        
        self.eval_stats = {
            "success_rate": success_rate,
            "avg_distance_to_goal_all": avg_distance_to_goal,
            "avg_distance_to_goal_success": avg_distance_to_goal_success,         
        }
        return


    def save_eval_results(self, model_dir: str, save_name: str) -> None:
        # save eval results in model yaml
        yaml_path = model_dir[:-3] + ".yaml"
        if not os.path.exists(yaml_path):
            return
        
        with open(yaml_path, "r") as file:
            data: dict = yaml.load(file, Loader=TrainCfgLoader)
        if "eval" not in data:
            data["eval"] = {}
                
        data["eval"][save_name] = self.eval_stats
        with open(yaml_path, "w") as file:
            yaml.dump(data, file)        
    
    ##
    # Plotting
    ##
    
    def plt_single_model(self, eval_dir: str, show: bool = True) -> None:
        # check if directory exists
        os.makedirs(eval_dir, exist_ok=True)
        
        # get goal success
        goal_success_bool = self.goal_distances < self.tolerance

        unique_goal_length = np.unique(np.round(self.length_goal, 1))
        goal_length_path_exists = []
        mean_path_extension = []
        std_path_extension = []
        mean_goal_distance = []
        std_goal_distance = []
        goal_counts = []
        for x in unique_goal_length:
            y_path_subset = ((self.length_path[goal_success_bool][np.round(self.length_goal[goal_success_bool], 1) == x] - x) / x) * 100   # deviation from goal length in percent for successful paths
            if len(y_path_subset) != 0:
                mean_path_extension.append(np.mean(y_path_subset))
                std_path_extension.append(np.std(y_path_subset))
                goal_length_path_exists.append(x)
            
            y_goal_subset = self.goal_distances[np.round(self.length_goal, 1) == x]
            mean_goal_distance.append(np.mean(y_goal_subset))
            std_goal_distance.append(np.std(y_goal_subset))
            goal_counts.append(len(y_goal_subset))

        ## plot with the distance to the goal depending on the length between goal and start
        avg_increase = np.mean((self.length_path / self.length_goal) -1)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle("Path Length Increase", fontsize=20)
        ax.plot(goal_length_path_exists, mean_path_extension, color='blue', label='Average path length')
        ax.fill_between(goal_length_path_exists, np.array(mean_path_extension) - np.array(std_path_extension), np.array(mean_path_extension) + np.array(std_path_extension), color='blue', alpha=0.2, label='Uncertainty')
        ax.set_xlabel('Start-Goal Distance', fontsize=16)
        ax.set_ylabel('Path Length', fontsize=16)
        ax.set_title(f"Avg increase of path length is {round(avg_increase, 5)*100:.2f}% for successful paths with tolerance of {self.tolerance}", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend()
        fig.savefig(os.path.join(eval_dir, "path_length.png"))
        if show:
            plt.show() 
        else:
            plt.close()

        ## plot to compare the increase in path length depending in on the distance between goal and start
        goal_success_mean = np.sum(goal_success_bool) / len(self.goal_distances)
        
        # Create a figure and two axis objects, with the second one sharing the x-axis of the first
        fig, ax1 = plt.subplots(figsize=(12, 10))
        ax2 = ax1.twinx()
        fig.subplots_adjust(hspace=0.4)  # Add some vertical spacing between the two plots

        # Plot the goal distance data
        ax1.plot(unique_goal_length, mean_goal_distance, color='blue', label='Average goal distance length', zorder=2)
        ax1.fill_between(unique_goal_length, np.array(mean_goal_distance) - np.array(std_goal_distance), np.array(mean_goal_distance) + np.array(std_goal_distance), color='blue', alpha=0.2, label='Uncertainty', zorder=1)
        ax1.set_xlabel('Start-Goal Distance', fontsize=16)
        ax1.set_ylabel('Goal Distance', fontsize=16)
        ax1.set_title(f"With a tolerance of {self.tolerance} are {round(goal_success_mean, 5)*100:.2f} % of goals reached", fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)

        # Plot the goal counts data on the second axis
        ax2.bar(unique_goal_length, goal_counts, color='red', alpha=0.5, width=0.05, label='Number of samples', zorder=0)
        ax2.set_ylabel('Sample count', fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=14)

        # Combine the legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        ax2.legend(lines+bars, labels+bar_labels, loc='upper center')

        plt.suptitle("Goal Distance", fontsize=20)
        fig.savefig(os.path.join(eval_dir, "goal_distance.png"))
        if show:
            plt.show() 
        else:
            plt.close()
        return
    
    def plt_comparison(
        self,
        length_path_list: List[np.ndarray],
        length_goal_list: List[np.ndarray],
        goal_distance_list: List[np.ndarray],
        model_dirs: List[str],
        save_dir: str,
    ) -> None:
        # path increase plot
        fig_path, axs_path = plt.subplots(figsize=(12, 10))
        fig_path.suptitle("Path Extension Comp", fontsize=20)
        axs_path.set_xlabel('Start-Goal Distance [m]', fontsize=16)
        axs_path.set_ylabel('Path Extension [%]', fontsize=16)         
        axs_path.tick_params(axis='both', which='major', labelsize=14)

        # goal distance plot
        fig_goal, axs_goal = plt.subplots(figsize=(12, 10))
        fig_goal.suptitle("Goal Distance Comp", fontsize=20)
        axs_goal.set_xlabel('Start-Goal Distance [m]', fontsize=16)
        axs_goal.set_ylabel('Goal Distance [m]', fontsize=16)
        axs_goal.tick_params(axis='both', which='major', labelsize=14)

        for idx in range(len(length_goal_list)):
            model_name = os.path.split(model_dirs[idx])[1]
            
            goal_success_bool = goal_distance_list[idx] < self.tolerance

            unique_goal_length = np.unique(np.round(length_goal_list[idx], 1))
            goal_length_path_exists = []
            mean_path_extension     = []
            std_path_extension      = []
            mean_goal_distance      = []
            std_goal_distance       = []
            for x in unique_goal_length:
                y_path_subset = ((length_path_list[idx][goal_success_bool][np.round(length_goal_list[idx][goal_success_bool], 1) == x] - x) / x) * 100
                if len(y_path_subset) != 0:
                    mean_path_extension.append(np.mean(y_path_subset))
                    std_path_extension.append(np.std(y_path_subset))
                    goal_length_path_exists.append(x)
                
                y_goal_subset = goal_distance_list[idx][np.round(length_goal_list[idx], 1) == x]
                mean_goal_distance.append(np.mean(y_goal_subset))
                std_goal_distance.append(np.std(y_goal_subset))

            ## plot to compare the increase in path length depending in on the distance between goal and start for the successful paths
            avg_increase = np.mean((length_path_list[idx] / length_goal_list[idx]) -1)
            axs_path.plot(goal_length_path_exists, mean_path_extension, label=f'{model_name} ({round(avg_increase, 5)*100:.2f} %))')
            axs_path.fill_between(goal_length_path_exists, np.array(mean_path_extension) - np.array(std_path_extension), np.array(mean_path_extension) + np.array(std_path_extension), alpha=0.2)

            ## plot with the distance to the goal depending on the length between goal and start
            goal_success = np.sum(goal_success_bool) / len(goal_distance_list[idx])        
            axs_goal.plot(unique_goal_length, mean_goal_distance, label=f'{model_name} ({round(goal_success, 5)*100:.2f} %)')
            axs_goal.fill_between(unique_goal_length, np.array(mean_goal_distance) - np.array(std_goal_distance), np.array(mean_goal_distance) + np.array(std_goal_distance), alpha=0.2)
        
        axs_path.legend()
        axs_goal.legend()
        fig_path.savefig(os.path.join(save_dir, "path_length_comp.png"))
        fig_goal.savefig(os.path.join(save_dir, "goal_distance_comp.png"))
        plt.show()
        return

# EoF
