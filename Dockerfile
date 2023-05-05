# Pull nivida cuda image
#   base: Includes the CUDA runtime (cudart)
#   runtime: Builds on the base and includes the CUDA math libraries, and NCCL. A runtime image that also includes cuDNN is available.
#   devel: Builds on the runtime and includes headers, development tools for building CUDA images. These images are particularly useful for multi-stage builds.
# if intended to compile CUDA code, use the devel image and make sure that the CUDA version matches the version used to compile pytorch (i.e. cu116 for nvidia/cuda:11.6.0)
# complete list of CUDA images https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md

#==
# FOUNDATION
#==
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,video,compute,utility

# Suppresses interactive calls to APT
ENV DEBIAN_FRONTEND="noninteractive"

# Install graphics drivers
RUN apt update && apt install -y libnvidia-gl-${DRIVER} \
  && rm -rf /var/lib/apt/lists/*

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
ENV TERM=xterm-256color

#==
# System APT base dependencies and utilities
#==

COPY bin/submodules/base.sh /home/base.sh
RUN chmod +x /home/base.sh
RUN /home/base.sh && rm /home/base.sh

#==
# ROS
#==

# Version
ARG ROS=noetic

COPY bin/submodules/ros_1.sh /home/ros_1.sh
RUN chmod +x /home/ros_1.sh
RUN /home/ros_1.sh && rm /home/ros_1.sh

#==
# JoyStick Planner DEPENDENCIES
#==
RUN apt install libusb-dev

#==
# VIPlanner DEPENDENCIES
#==
RUN pip install wheel
RUN pip install torch>=1.13.1
RUN pip install torchvision>=0.14.1
RUN pip install torchaudio>=0.13.1
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install trimesh

#==
# VIPlanner
#==
COPY planner/model_src/viplanner /viplanner
COPY planner/model_src/viplanner/viplanner/third_party/mask2former /mask2former

# install mask2former
RUN pip install -r /mask2former/requirements.txt
RUN chmod 777 '/usr/local/lib/python3.8/dist-packages'
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH
ENV FORCE_CUDA="1"
RUN pip install ninja
RUN python3 /mask2former/mask2former/modeling/pixel_decoder/ops/setup.py build install

# install viplanner in edible mode (needed update of pip and setuptools) --> update viplanner without rebuilding the image
RUN pip install --upgrade pip
RUN pip install setuptools==66.0.0
# FIX for PyYAML 6.0.0 install error (see README.md)
RUN pip install --ignore-installed PyYAML==6.0.0
RUN pip install -e /viplanner/.[sim]

#==
# Cleanup
#==
RUN apt update && apt upgrade -y

#==
# Environment
#==
COPY bin/submodules/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

#==
# Execution
#==
COPY bin/entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD []
