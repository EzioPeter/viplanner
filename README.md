# VIPlanner: Visual-Imperative-Planner

VIPlanner is a robust learning-based local path planner based on RGB and depth images.
Fully trained in simulation, the planner can be applied in with additional training in dynamic indoor as well outdoor environments.
We provide it as an extenstion for [NVIDIA Isaac-Sim](https://developer.nvidia.com/isaac-sim) within the [Orbit](https://github.com/leggedrobotics/orbit/tree/dev/pascal/anymal-vip) project.
Furthermore, a ready to use [ROS Noetic](http://wiki.ros.org/noetic) package is available [here](https://github.com/pascal-roth/viplanner_ros) for direct integration on any robot (tested and developed on ANYmal D). 

**Main Contact:** Pascal Roth ([rothpa@ethz.ch](mailto:rothpa@ethz.ch?subject=[GitHub]))

**Authors:** [Pascal Roth](https://github.com/pascal-roth), [Fan Yang](https://github.com/MichaelFYang), [Julian Nubert](https://juliannubert.com/), [Marco Hutter](https://rsl.ethz.ch/the-lab/people/person-detail.MTIxOTEx.TGlzdC8yNDQxLC0xNDI1MTk1NzM1.html)



## Install

TODO: formulate properly

- Install 'pyproject.toml' with pip by running: ```pip install .``` or ```pip install -e .``` if you want to edit the code
- Manually install detectron2 'pip install ```git+https://github.com/facebookresearch/detectron2.git'```
- Build mask2former cuda operators 
```
	cd third_party/mask2former/mask2former/modeling/pixel_decoder/ops
	sh make.sh
```

**Remark**
Note that for an ediable install for packages without setup.py, PEP660 has to be implemented. This requires the following versions (as described [here](https://stackoverflow.com/questions/69711606/how-to-install-a-package-using-pip-in-editable-mode-with-pyproject-toml))
- [pip >= 21.3](https://pip.pypa.io/en/stable/news/#v21-3)
- [setuptools >= 64.0.0](https://github.com/pypa/setuptools/blob/main/CHANGES.rst#v6400)


## Usage


### Training


### Inference

