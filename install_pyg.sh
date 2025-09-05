#!/bin/bash

# 配置项（可修改）
ENV_NAME=pyg_env
PYTHON_VERSION=3.10
TORCH_VERSION=2.1.0
CUDA_VERSION=cu118
ALIYUN_WHL=https://mirrors.aliyun.com/pytorch-wheels/$CUDA_VERSION
PYG_WHL=https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

echo "🚀 创建 Conda 环境：$ENV_NAME (Python $PYTHON_VERSION)"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "⚙️ 使用阿里云镜像安装 PyTorch $TORCH_VERSION ($CUDA_VERSION)"
pip install torch==$TORCH_VERSION+${CUDA_VERSION} torchvision==0.16.2+${CUDA_VERSION} torchaudio==2.1.0+${CUDA_VERSION} -f $ALIYUN_WHL

echo "🔧 安装 PyG 依赖（使用匹配的 .whl 包）"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f $PYG_WHL

echo "📦 安装主包 torch-geometric"
pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "🧪 写入测试文件 test_pyg.py"
cat <<EOF > test_pyg.py
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print('✅ Dataset:', dataset)
print('✅ Sample:', dataset[0])
EOF

echo "🚀 运行测试脚本"
python test_pyg.py

echo "✅ 安装完成！环境名: $ENV_NAME"
