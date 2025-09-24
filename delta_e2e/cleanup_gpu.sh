#!/bin/bash

echo "清理GPU进程..."

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi 不可用，跳过GPU进程清理"
    exit 0
fi

# 获取当前用户的GPU进程
echo "当前GPU进程："
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits

# 终止所有GPU进程
echo "正在终止GPU进程..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | while read pid; do
    if [ ! -z "$pid" ]; then
        echo "终止进程 $pid"
        kill -TERM "$pid" 2>/dev/null || true
    fi
done

# 等待2秒
sleep 2

# 强制终止仍在运行的GPU进程
echo "强制终止剩余GPU进程..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | while read pid; do
    if [ ! -z "$pid" ]; then
        echo "强制终止进程 $pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
done

echo "GPU进程清理完成"
