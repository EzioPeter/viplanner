# python
import argparse
import os
from typing import List, Optional, Union

import numpy as np
import torch

# imperative-planning-learning
from viplanner.config import TrainCfg
from viplanner.traj_cost_opt import TrajCost
from viplanner.utils.eval_utils import BaseEvaluator
from viplanner.utils.trainer import Trainer

DATA_PARENT_DIR = "/home/pascal/viplanner/imperative_learning/data"


class SimEvaluator(BaseEvaluator):
    debug: bool = False

    def __init__(
        self,
        distance_tolerance: float = 0.5,
        obs_loss_threshold: float = 0.3,
        environment: Union[List[str], str] = "2n8kARJN3HM",
        carla: bool = False,
        fear_filter: bool = False,
    ) -> None:
        """Evaluation of Simulation Results with one-shot predictions

        Args:
            tolerance (float, optional): Tolerance to classify a goal as reached. Defaults to 0.5.
            environment (str, optional): Name of the environment used for testing. Defaults to "2n8kARJN3HM".
            carla (bool, optional): If it is a Carla Simulator environment (affects the DataCfg). Defaults to False.
        """

        # init base class
        super().__init__(
            distance_tolerance=distance_tolerance,
            obs_loss_threshold=obs_loss_threshold,
        )
        # parameters
        self.carla: bool = carla
        self.fear_filter: bool = fear_filter
        # set random seed for reproducibility
        torch.manual_seed(12)
        # environment
        self.environment = environment
        # buffer
        self.trainer: Trainer = None
        self._traj_cost: TrajCost = None
        return

    def run(
        self,
        model_dirs: List[str],
        model_names: Optional[List[str]] = None,
        use_prev_results: bool = True,
    ):
        # check if enough environments are given
        if isinstance(self.environment, list):
            assert len(self.environment) == len(model_dirs), (
                "Number of environments and models must match to assign each"
                " model its own environment!"
            )

        # init buffers
        length_goal_list = []
        length_path_list = []
        path_extension_list = []
        goal_distance_list = []
        obstacle_loss_list = []
        # obstacle_max_loss_list = []

        for idx, model_dir in enumerate(model_dirs):
            _, model_name = os.path.split(model_dir)

            # check for previous results
            if use_prev_results:
                # check if all file exist
                if isinstance(self.environment, list):
                    data_dir = os.path.join(
                        DATA_PARENT_DIR, self.environment[idx]
                    )
                else:
                    data_dir = os.path.join(DATA_PARENT_DIR, self.environment)
                eval_dir = os.path.join(data_dir, f"eval_{model_name}")
                all_file_exists = all(
                    [
                        os.path.isfile(
                            os.path.join(eval_dir, f"{file_name}.txt")
                        )
                        for file_name in [
                            "length_goal",
                            "length_path",
                            "path_extension",
                            "goal_distances",
                            "loss_obstacles",
                        ]
                    ]
                )

                if all_file_exists:
                    length_goal_list.append(
                        np.loadtxt(os.path.join(eval_dir, "length_goal.txt"))
                    )
                    length_path_list.append(
                        np.loadtxt(os.path.join(eval_dir, "length_path.txt"))
                    )
                    path_extension_list.append(
                        np.loadtxt(
                            os.path.join(eval_dir, "path_extension.txt")
                        )
                    )
                    goal_distance_list.append(
                        np.loadtxt(
                            os.path.join(eval_dir, "goal_distances.txt")
                        )
                    )
                    obstacle_loss_list.append(
                        np.loadtxt(
                            os.path.join(eval_dir, "loss_obstacles.txt")
                        )
                    )
                    self._use_cost_map = True
                    # obstacle_max_loss_list.append(np.loadtxt(os.path.join(eval_dir, "loss_max_obstacles.txt")))
                    continue
                else:
                    print(
                        "[INFO] No previous results found, running"
                        " evaluation..."
                    )

            # load trainer, config and data
            if isinstance(self.environment, list):
                train_cfg = self.load_config(model_dir, idx=idx)
                if idx == 0 or (
                    idx > 0
                    and self.environment[idx] != self.environment[idx - 1]
                ):
                    self.trainer = Trainer(
                        train_cfg
                    )  # can get new trainer since data and cost changes
                    self.setup_data_cost()  # load individual data and cost
                else:
                    # load new config
                    self.trainer._cfg = train_cfg
            else:
                train_cfg = self.load_config(model_dir)
                if self.trainer is None:
                    # load trainer
                    self.trainer = Trainer(train_cfg)
                    # load data
                    self.setup_data_cost()
                else:
                    # load new config
                    self.trainer._cfg = train_cfg

            # reset
            self.reset()
            del self.trainer.net

            # load model
            self.trainer.model_path = os.path.join(model_dir, "model.pt")
            self.trainer._load_model(resume=True)

            # run evaluation
            self.run_eval(model_name)

            length_goal_list.append(self.length_goal.copy())
            length_path_list.append(self.length_path.copy())
            goal_distance_list.append(self.goal_distances.copy())
            obstacle_loss_list.append(self.loss_obstacles.copy())
            path_extension_list.append(self.path_extension.copy())

        self.plt_comparison(
            length_goal_list=length_goal_list,
            goal_distance_list=goal_distance_list,
            path_extension_list=path_extension_list,
            model_dirs=model_dirs,
            save_dir=(
                os.path.join(DATA_PARENT_DIR, self.environment[0])
                if isinstance(self.environment, list)
                else os.path.join(DATA_PARENT_DIR, self.environment)
            ),
            obs_loss_list=obstacle_loss_list,
            model_names=model_names,
        )
        return

    def load_config(self, model_dir: str, idx: int = 0) -> TrainCfg:
        # load config
        train_config: TrainCfg = TrainCfg.from_yaml(
            os.path.join(model_dir, "model.yaml")
        )
        # set environment
        if isinstance(self.environment, list):
            train_config.env_list = [self.environment[idx]]
        else:
            train_config.env_list = [self.environment]
        # set data config
        if self.carla:
            if isinstance(train_config.data_cfg, list):
                carla_idx = [
                    data_cfg.carla for data_cfg in train_config.data_cfg
                ].index(True)
                if carla_idx:
                    print(
                        "[INFO] Carla data found, only using first data"
                        f" config: {train_config.data_cfg[carla_idx]}"
                    )
                    train_config.data_cfg = [
                        train_config.data_cfg[carla_idx[0]]
                    ]
                else:
                    print(
                        "[WARNING] No Carla data found, using first data"
                        f" config: {train_config.data_cfg[0]}"
                    )
                    train_config.data_cfg = [train_config.data_cfg[0]]
            else:
                if train_config.data_cfg.carla:
                    print(f"[INFO] Carla data found: {train_config.data_cfg}")
                    train_config.data_cfg = [train_config.data_cfg]
                else:
                    print(
                        "[WARNING] No Carla data found, using data config:"
                        f" {train_config.data_cfg}"
                    )
                    train_config.data_cfg = [train_config.data_cfg]
        else:
            if isinstance(train_config.data_cfg, list):
                print(
                    "[INFO] Using first data config:"
                    f" {train_config.data_cfg[0]}"
                )
                train_config.data_cfg = [train_config.data_cfg[0]]
            else:
                train_config.data_cfg = [train_config.data_cfg]
        # data should be 100% within the fov (otherwise only results in )
        train_config.data_cfg[0].ratio_fov_samples = 1.0
        train_config.data_cfg[0].ratio_front_samples = 0.0
        train_config.data_cfg[0].ratio_back_samples = 0.0
        # enforce that this environment is used for testing
        train_config.test_env_id = 0
        return train_config

    def setup_data_cost(self):
        # get dataloader for training
        self.trainer._load_data(train=False)
        _, test_loader = self.trainer._get_dataloader(
            train=False, allow_augmentation=False
        )
        self.test_loader = test_loader[0]

        # set cost map
        if self._traj_cost is None:
            self._use_cost_map = True
            self._traj_cost: TrajCost = self.trainer.data_traj_cost[0]

        # create buffers
        nbr_samples = len(self.test_loader) * self.trainer._cfg.batch_size
        self.set_nbr_paths(nbr_samples)
        return

    def create_buffers(self) -> None:
        super().create_buffers()
        self.loss_max_obstacles = np.zeros(self.nbr_paths)
        return

    def run_eval(self, model_name: str) -> None:
        pred_counter = 0

        with torch.no_grad():
            for inputs in self.test_loader:
                odom = inputs[2].cuda(self.trainer._cfg.gpu_id)
                goal = inputs[3].cuda(self.trainer._cfg.gpu_id)

                if self.trainer._cfg.sem or self.trainer._cfg.rgb:
                    image = inputs[0].cuda(self.trainer._cfg.gpu_id)  # depth
                    sem_rgb_image = inputs[1].cuda(
                        self.trainer._cfg.gpu_id
                    )  # semantic
                    preds, fear = self.trainer.net(image, sem_rgb_image, goal)
                else:
                    image = inputs[0].cuda(self.trainer._cfg.gpu_id)
                    preds, fear = self.trainer.net(image, goal)

                # flip y axis for augmented samples
                preds[inputs[4], :, 1] = preds[inputs[4], :, 1] * -1
                goal[inputs[4], 1] = goal[inputs[4], 1] * -1

                # filter paths with high fear
                if self.fear_filter:
                    fear_selection = (fear < 0.5).squeeze()
                    preds = preds[fear_selection]
                    goal = goal[fear_selection]
                    odom = odom[fear_selection]
                    fear = fear[fear_selection]

                if len(preds.shape) == 4:
                    # squeeze
                    preds = preds.squeeze(0)
                    goal = goal.squeeze(0)
                    odom = odom.squeeze(0)
                    fear = fear.squeeze(0)

                # optimize
                waypoints = self._traj_cost.opt.TrajGeneratorFromPFreeRot(
                    preds, step=0.1
                )
                waypoints_world = (
                    self._traj_cost.TransformPoints(odom, waypoints)
                    .tensor()
                    .cpu()
                    .numpy()
                )
                goal_world = (
                    self._traj_cost.TransformPoints(odom, goal[:, None, :3])
                    .tensor()
                    .cpu()
                    .numpy()
                )
                # evaluate
                self.goal_distances[
                    pred_counter : pred_counter + len(goal)
                ] = np.linalg.norm(
                    waypoints_world[:, -1, :2] - goal_world[:, 0, :2], axis=1
                )
                self.length_path[pred_counter : pred_counter + len(goal)] = (
                    np.sum(
                        np.linalg.norm(
                            waypoints_world[:, 1:, :2]
                            - waypoints_world[:, :-1, :2],
                            axis=2,
                        ),
                        axis=1,
                    )
                )
                self.length_goal[pred_counter : pred_counter + len(goal)] = (
                    np.linalg.norm(
                        goal_world[:, 0, :2] - odom[:, :2].cpu().numpy(),
                        axis=1,
                    )
                )
                mean_loss, max_loss = self._traj_cost.obs_cost_eval(
                    odom, waypoints
                )
                self.loss_obstacles[
                    pred_counter : pred_counter + len(goal)
                ] = mean_loss.cpu().numpy()
                self.loss_max_obstacles[
                    pred_counter : pred_counter + len(goal)
                ] = max_loss.cpu().numpy()
                # path extension as (path length - straight line start to final point on path) / (straight line start to final point on path)
                straight_distance = np.linalg.norm(
                    waypoints_world[:, -1, :2] - odom[:, :2].cpu().numpy(),
                    axis=1,
                )
                self.path_extension[
                    pred_counter : pred_counter + len(goal)
                ] = (
                    self.length_path[pred_counter : pred_counter + len(goal)]
                    - straight_distance
                ) / straight_distance
                pred_counter += len(goal)

                if self.debug:
                    path_diff = (
                        self.length_path[
                            pred_counter : pred_counter + len(goal)
                        ]
                        - self.length_goal[
                            pred_counter : pred_counter + len(goal)
                        ]
                    )
                    largest_indices = torch.tensor(np.argsort(path_diff)[-30:])
                    self.trainer.data_traj_viz[0].VizTrajectory(
                        preds.cpu()[largest_indices],
                        waypoints.cpu()[largest_indices],
                        odom.cpu()[largest_indices],
                        goal.cpu()[largest_indices],
                        fear.cpu()[largest_indices],
                        fov_angle=self.trainer.data_generators[0].alpha_fov,
                        augment_viz=inputs[4].cpu()[largest_indices],
                    )

        # crop buffers
        self.goal_distances = self.goal_distances[:pred_counter]
        self.length_path = self.length_path[:pred_counter]
        self.length_goal = self.length_goal[:pred_counter]
        self.loss_obstacles = self.loss_obstacles[:pred_counter]
        self.loss_max_obstacles = self.loss_max_obstacles[:pred_counter]
        self.path_extension = self.path_extension[:pred_counter]

        # sort values
        sort_indices = np.argsort(self.length_goal)
        self.length_goal = self.length_goal[sort_indices]
        self.length_path = self.length_path[sort_indices]
        self.goal_distances = self.goal_distances[sort_indices]
        self.loss_obstacles = self.loss_obstacles[sort_indices]
        self.loss_max_obstacles = self.loss_max_obstacles[sort_indices]
        self.path_extension = self.path_extension[sort_indices]

        # make directory and save data
        data_dir = os.path.join(
            self.trainer._cfg.data_dir,
            self.trainer._cfg.env_list[self.trainer._cfg.test_env_id],
        )
        eval_dir = os.path.join(data_dir, f"eval_{model_name}")
        os.makedirs(eval_dir, exist_ok=True)

        np.savetxt(os.path.join(eval_dir, "length_path.txt"), self.length_path)
        np.savetxt(os.path.join(eval_dir, "length_goal.txt"), self.length_goal)
        np.savetxt(
            os.path.join(eval_dir, "goal_distances.txt"), self.goal_distances
        )
        np.savetxt(
            os.path.join(eval_dir, "loss_obstacles.txt"), self.loss_obstacles
        )
        np.savetxt(
            os.path.join(eval_dir, "loss_max_obstacles.txt"),
            self.loss_max_obstacles,
        )
        np.savetxt(
            os.path.join(eval_dir, "path_extension.txt"), self.path_extension
        )

        # plot data
        self.plt_single_model(eval_dir, show=False)

        # get statistics
        self.eval_statistics()
        self.save_eval_results(
            self.trainer._cfg.curr_model_dir,
            save_name=os.path.split(data_dir)[-1],
        )
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Model Eval", description="Evaluate VIPmodels"
    )
    parser.add_argument(
        "-m",
        "--model_dirs",
        nargs="+",
        type=str,
        help="Path to model directory",
        default=[
            "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_new_colorspace_ep100_inputDepSem_costSem_optimSGD_new_colorspace_sharpend_indoor",
            "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD_new_loss_neg05",
            "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD_neg05",
            # "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD_combi_more_data_neg05",
            # "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDep_costSem_optimSGD_depth",
        ],
    )
    parser.add_argument(
        "-n",
        "--model_names",
        nargs="+",
        type=str,
        help="Model name",
        default=[
            "VIPlanner (rectangle approx with new loss and new colorspace)",
            "VIPlanner (rectangle approx)",
            "VIPlanner (old)",
            # "iPlanner",
        ],
    )
    parser.add_argument(
        "-env",
        "--environment",
        type=str,
        help="Environment name",
        default=[
            "2n8kARJN3HM_new_colorspace",
            "2n8kARJN3HM",
            "2n8kARJN3HM",
        ],
    )  # "town01_more_data_train")
    parser.add_argument(
        "-c",
        "--carla",
        action="store_true",
        help="Use carla environment (changes DataCfg)",
        default=False,
    )
    parser.add_argument(
        "-ff",
        "--fear_filter",
        action="store_true",
        help=(
            "Filter all fear trajectories for evaluation (filtered fear values"
            " above 0.5)"
        ),
        default=True,
    )
    parser.add_argument(
        "--distance_tolerance",
        type=float,
        help="Tolerance to the goal to be considered reached",
        default=0.5,
    )
    parser.add_argument(
        "--obs_loss_threshold",
        type=float,
        help="Maximum obstacle loss for a path to be considered successful",
        default=0.3,
    )
    args = parser.parse_args()
    print(args)

    evaluator = SimEvaluator(
        args.distance_tolerance,
        args.obs_loss_threshold,
        args.environment,
        args.carla,
        args.fear_filter,
    )
    evaluator.run(args.model_dirs, args.model_names)

# EoF
