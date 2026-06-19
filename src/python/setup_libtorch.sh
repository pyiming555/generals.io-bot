#!/bin/bash
set -e
cd /media/pyiming/C22AA0E82AA0DB25/project/generals.io

if [ ! -d libtorch ]; then
    echo "下载 LibTorch 2.4.1 (CPU)..."
    wget -q --show-progress https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip -O libtorch.zip
    echo "解压..."
    unzip -q libtorch.zip
    rm libtorch.zip
    echo "LibTorch 就绪: $(ls libtorch/include/torch/csrc/api/include/torch/torch.h)"
else
    echo "LibTorch 已存在，跳过下载"
fi

# 检查 torch.h 是否存在
ls libtorch/include/torch/csrc/api/include/torch/torch.h
