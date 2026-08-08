# syntax=docker/dockerfile:1.7

# ---- ベース共通部分 ----
ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build libgl1 libglib2.0-0 ffmpeg \
    v4l-utils \
    libxcb1 libglu1-mesa libx11-xcb1 libxrender1 \
    libsm6 libice6 libxext6 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install "numpy<2.0" && \
    pip install -U openmim && \
    pip install "mmengine>=0.10.4" "mmdet>=3.2.0,<3.3.0"

# ---- 公式安定版(mmcv prebuilt wheelがある組み合わせ) ----
FROM base AS stable
RUN mim install "mmcv==2.1.0"
RUN mim install "mmpose>=1.3.0" && \
    pip install "numpy<2.0" --force-reinstall --no-deps

# ---- サードパーティ最新版(miropsota, RTX50xx対応) ----
FROM base AS thirdparty-latest
RUN pip install mmcv==2.2.0+pt2.7.0cu128 \
    --extra-index-url https://miropsota.github.io/torch_packages_builder
# mmdetのmmcv上限チェックを緩和(既知の未マージ修正の代替)
RUN sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
    /opt/conda/lib/python3.11/site-packages/mmdet/__init__.py
RUN mim install "mmpose>=1.3.0" && \
    pip install "numpy<2.0" --force-reinstall --no-deps
    
# ---- ソースビルド版(prebuiltが無い組み合わせの保険) ----
FROM base AS source-build
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"
RUN pip install "mmcv==2.2.0" --no-binary mmcv
RUN mim install "mmpose>=1.3.0" && \
    pip install "numpy<2.0" --force-reinstall --no-deps

# ---- CPUのみ版 ----
FROM pytorch/pytorch:2.7.0-cpu AS cpu-only
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U openmim \
    && pip install "mmengine>=0.10.4" "mmdet>=3.2.0,<3.3.0" \
    && mim install "mmcv==2.1.0" \
    && mim install "mmpose>=1.3.0" \
    && pip install "numpy<2.0" --force-reinstall --no-deps
