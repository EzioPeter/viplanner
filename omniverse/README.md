# ViPlanner Omniverse Extension

The ViPlanner Omniverse Extension offers a sophisticated testing environment for ViPlanner.
Within NVIDIA Isaac Sim as a photorealistic simulator, this extension provides an assessment tool for ViPlanner's performance across diverse environments.
The extension is developed using the [Orbit Framework](https://isaac-orbit.github.io/).

**Remark**

The extension for `Matterport` and `Unreal Engine` meshes with semantic information is currently getting updated to the latest Orbit version and will be available soon. An intermediate solution is given [here](https://github.com/pascal-roth/orbit_envs).

## Installation

To install the ViPlanner extension for Isaac Sim, follow these steps:

1. Install Isaac Sim using the [Orbit installation guide](https://isaac-orbit.github.io/orbit/source/setup/installation.html).
2. Clone the orbit repo and link the viplanner extension.

```
git clone git@github.com:NVIDIA-Omniverse/orbit.git
cd orbit/source/extension
ln -s {VIPLANNER_DIR}/omniverse/extension/omni.viplanner .
```

3. TEMPORARY: To use Matterport and Unreal Engine Meshes with semantic information within Isaac Sim, a new extension has been developed as part of this work. Currently, all parts are getting updated to the latest Orbit version. A temporary solution that is sufficient for the demo script is available [here](https://github.com/pascal-roth/orbit_envs). Please also clone and link it into orbit.

```
git clone git@github.com:pascal-roth/orbit_envs.git
cd orbit/source/extension
ln -s {ORBIT_ENVS}/extensions/omni.isaac.matterport .
ln -s {ORBIT_ENVS}/extensions/omni.isaac.carla .
```

4. Then run the orbit installer script and additionally install ViPlanner in the Isaac Sim virtual environment.

```
./orbit.sh -i -e
./orbit.sh -p -m pip install -e {VIPLANNER_DIR}
```

**Remark**
Also in orbit, it is necessary to comply with PEP660 for the install. This requires the following versions (as described [here](https://stackoverflow.com/questions/69711606/how-to-install-a-package-using-pip-in-editable-mode-with-pyproject-toml) in detail)
- [pip >= 21.3](https://pip.pypa.io/en/stable/news/#v21-3)
	```
  ./orbit.sh -p -m pip install --upgrade pip
  ```
- [setuptools >= 64.0.0](https://github.com/pypa/setuptools/blob/main/CHANGES.rst#v6400)
	```
  ./orbit.sh -p -m pip install --upgrade setuptools
  ```

## Usage

A demo script is provided to run the planner in three different environments: [Matterport](https://niessner.github.io/Matterport/), [Carla](https://carla.org//), and [NVIDIA Warehouse](https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_environments.html#warehouse).
In each scenario, the goal is represented as a movable cube within the environment.

### Matterport
[Download USD Link](https://drive.google.com/file/d/1BZBRApnfizoUdOrsihinMD12RQk9G1CK/view?usp=sharing) [Download PLY Link](https://drive.google.com/file/d/1_jgpM-xRvFOMH1C78IgDDiyEtt9hKauz/view?usp=sharing)
```
./orbit.sh -p {VIPLANNER_DIR}/omniverse/standalone/viplanner_demo.py --scene matterport
```

### Carla
[Download Mesh Link](https://drive.google.com/file/d/1_jgpM-xRvFOMH1C78IgDDiyEtt9hKauz/view?usp=sharing)  [Download Texture Link](https://drive.google.com/file/d/1jsvkObiLOwg_zoVTC4vO7JprETSHdb7N/view?usp=sharing)
```
./orbit.sh -p {VIPLANNER_DIR}/omniverse/standalone/viplanner_demo.py --scene matterport
```

### NVIDIA Warehouse
```
./orbit.sh -p {VIPLANNER_DIR}/omniverse/standalone/viplanner_demo.py --scene matterport
```

## Data Collection and Evaluation

Script for data collection and evaluation are getting updated to the latest Orbit version and will be available soon. If you are interested in the current state, please contact us.
