FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu24.04 AS toolkit_source
FROM nvcr.io/nvidia/isaac-sim:5.1.0

# COPY --from=isaac-sim /isaac-sim /isaac-sim

# Environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

USER root

RUN apt-get update && apt-get install -y git-lfs

ENV omni_python='/isaac-sim/python.sh'

RUN echo "alias omni_python='/isaac-sim/python.sh'" >> ~/.bashrc

RUN mkdir /pkgs && cd /pkgs && git clone https://github.com/NVlabs/curobo.git


RUN apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3-dev


ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"

WORKDIR /isaac-sim

COPY --from=toolkit_source /usr/local/cuda-12.8 /usr/local/cuda-12.8
RUN ln -s /usr/local/cuda-12.8 /usr/local/cuda
ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV CUDA_HOME=/usr/local/cuda
RUN nvcc --version
# WORKDIR /pkgs/curobo

WORKDIR /root
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN ls

ENV CONDA_DIR=/root/conda
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b -p $CONDA_DIR
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN /root/conda/bin/conda init bash

RUN mamba create -n curobo python=3.11 pip -y
RUN mamba shell init --shell bash

RUN echo "mamba activate curobo" >> ~/.bashrc


WORKDIR /pkgs/curobo

# use bash as the default shell

RUN /envs/curobo/bin/python --version && /envs/curobo/bin/pip --version
RUN /envs/curobo/bin/python --version && /envs/curobo/bin/pip --version

RUN --mount=type=cache,target=/root/.cache/pip /envs/curobo/bin/python -m pip install ninja wheel tomli warp-lang 
RUN --mount=type=cache,target=/root/.cache/pip /envs/curobo/bin/python -m pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com


RUN --mount=type=cache,target=/root/.cache/pip /envs/curobo/bin/python -m pip install -e .[dev] --no-build-isolation

