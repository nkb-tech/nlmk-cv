# syntax=docker/dockerfile:1

# Does not work because of RTSP connection
# FROM nvcr.io/nvidia/pytorch:23.09-py3 AS base

# Variables used at build time.
## Base CUDA version. See all supported version at https://hub.docker.com/r/nvidia/cuda/tags?page=2&name=-devel-ubuntu
ARG CUDA_VERSION=11.8.0
## Base Ubuntu version.
ARG OS_VERSION=22.04

# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION} AS base

# Dublicate args because of the visibility zone
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CUDA_VERSION
ARG OS_VERSION

## Base TensorRT version.
ARG TRT_VERSION=8.6.1.6
## Base PyTorch version.
ARG TORCH_VERSION=2.1.0
## Base Timezone
ARG TZ=Europe/Moscow

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive \
    ## Set timezone as it is required by some packages.
    TZ=${TZ} \
    ## CUDA Home, required to find CUDA in some packages.
    CUDA_HOME="/usr/local/cuda" \
    ## Set LD_LIBRARY_PATH for local libs (glog etc.)
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    ## Accelerate compilation flags (use all cores)
    MAKEFLAGS=-j$(nproc)

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
# openssl and tar due to security updates https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt update && \
    apt install \
        --no-install-recommends \
        --yes \
            build-essential \
            cmake \
            ca-certificates \
            git \
            git-lfs \
            zip \
            unzip \
            curl \
            wget \
            htop \
            libgl1 \
            libglib2.0-0 \
            gnupg \
            libusb-1.0-0 \
            openssl \
            tar \
            tzdata \
            python-is-python3 \
            python3.10-dev \
            python3-pip \
            ffmpeg && \
    ## Clean cached files
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    ## Set timezone
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

SHELL ["/bin/bash", "-c"]

# Install TensorRT
## Now only supported for Ubuntu 22.04
## Cannot install via pip because cuda-based errors
RUN v="${TRT_VERSION}-1+cuda${CUDA_VERSION%.*}" distro="ubuntu${OS_VERSION//./}" arch=$(uname -m) && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/cuda-archive-keyring.gpg && \
    mv cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/ /" | \
    tee /etc/apt/sources.list.d/cuda-${distro}-${arch}.list && \
    apt-get update && \
    apt-get install \
        libnvinfer-headers-dev=${v} \
        libnvinfer-dispatch8=${v} \
        libnvinfer-lean8=${v} \
        libnvinfer-dev=${v} \
        libnvinfer-headers-plugin-dev=${v} \
        libnvinfer-lean-dev=${v} \
        libnvinfer-dispatch-dev=${v} \
        libnvinfer-plugin-dev=${v} \
        libnvinfer-vc-plugin-dev=${v} \
        libnvparsers-dev=${v} \
        libnvonnxparsers-dev=${v} \
        libnvinfer8=${v} \
        libnvinfer-plugin8=${v} \
        libnvinfer-vc-plugin8=${v} \
        libnvparsers8=${v} \
        libnvonnxparsers8=${v} && \
    apt-get install \
        python3-libnvinfer=${v} \
        tensorrt-dev=${v} && \
    apt-mark hold tensorrt-dev

# Install pip packages
RUN python3 -m pip install \
    --upgrade \
        pip \
        wheel \
        setuptools \
        ninja && \
    ## Install pytorch and submodules
    CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python3 -m pip install \
        torch==${TORCH_VERSION}+cu${CUDA_VER} \
            --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

# Create working directory
WORKDIR /usr/src/

# Install nkb-tech detection pipeline
RUN git clone https://github.com/nkb-tech/ultralytics.git -b main /usr/src/ultralytics && \
    cd ultralytics && \
    python3 -m pip install \
        --no-cache \
        --editable \
            ".[export]" \
            albumentations \
            comet \
            pycocotools \
            pytest-cov && \
    # Due to error when installing albumentations
    # module 'cv2.dnn' has no attribute 'DictValue'
    # Downgrade opencv version to 4.8.0.74
    python3 -m pip uninstall \
        --yes \
            opencv-python-headless \
            opencv-python && \
    python3 -m pip install \
        --no-cache \
        opencv-python==4.8.0.74 && \
    # Run exports to AutoInstall packages
    yolo export model=tmp/yolov8n.pt format=edgetpu imgsz=32 && \
    yolo export model=tmp/yolov8n.pt format=ncnn imgsz=32 && \
    python3 -m pip install install \
        --no-cache \
            # Requires <= Python 3.10, bug with paddlepaddle==2.5.0 https://github.com/PaddlePaddle/X2Paddle/issues/991
            paddlepaddle==2.4.2 \
            x2paddle \
            # Fix error: `np.bool` was a deprecated alias for the builtin `bool` segmentation error in Tests
            numpy==1.23.5 && \
    rm -rf tmp

# Install nkb-tech cv pipeline
# Use copy because of it is private project
COPY . /usr/src/app
RUN cd /usr/src/app && \
    python3 -m pip install \
        --no-cache \
        --requirement \
            requirements.txt
