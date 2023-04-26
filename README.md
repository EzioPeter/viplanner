# Visual-Imperative-Planner (VIPlanner)


## imperative_cost_map
Design of differentiable cost-maps for Imperative Learning

Required submodules:
- Image Processing for Basic Depth Completion (IP-Basic), found [here](https://github.com/kujason/ip_basic)


# Install

TODO: formulate properly

- Install 'pyproject.toml' with pip by running: ```pip install .```
- Manually install detectron2 'pip install ```git+https://github.com/facebookresearch/detectron2.git'```
- Build mask2former cuda operators 
```
	cd third_party/mask2former/mask2former/modeling/pixel_decoder/ops
	sh make.sh
```

