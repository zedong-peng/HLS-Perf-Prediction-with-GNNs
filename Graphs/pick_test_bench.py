# input是designs dir. in the designs dir there are : source_name/kernel_name/design_id
# in each source_name dir, pick 10% kernels and mv total kernels folder to path/to/test_bench

import os
import random
import shutil
import argparse
import sys
from datetime import datetime

def pick_test_bench(designs_dir, test_bench_dir, test_ratio=0.1, save_log=True):
    """
    从designs_dir中随机选择test_ratio比例的内核，并将它们移动到test_bench_dir，
    使用扁平化的目录结构：test_bench/kernel_name。
    
    Args:
        designs_dir: 设计文件的源目录
        test_bench_dir: 测试基准的目标目录
        test_ratio: 选择作为测试集的内核比例（默认为0.1，即10%）
        save_log: 是否保存选择的内核信息到日志文件（默认为True）
    """
    if not os.path.exists(test_bench_dir):
        os.makedirs(test_bench_dir)
    
    # 用于记录所有选择的内核
    selected_kernels = []
    
    for source_name in os.listdir(designs_dir):
        source_path = os.path.join(designs_dir, source_name)
        if not os.path.isdir(source_path):
            continue
            
        # 获取所有内核名称
        kernel_names = [k for k in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, k))]
        
        # 随机选择test_ratio比例的内核
        num_test_kernels = max(1, int(len(kernel_names) * test_ratio))
        test_kernels = random.sample(kernel_names, num_test_kernels)
        
        # 记录选择的内核
        for kernel in test_kernels:
            selected_kernels.append(f"{source_name}/{kernel}")
        
        print(f"从{source_name}中选择了{len(test_kernels)}/{len(kernel_names)}个内核作为测试集")
        
        for kernel_name in test_kernels:
            kernel_path = os.path.join(source_path, kernel_name)
            # 修改目标路径，直接使用kernel_name作为目录名
            target_kernel_path = os.path.join(test_bench_dir, kernel_name)
            
            # 如果目标目录已存在，添加源名称作为后缀以避免冲突
            if os.path.exists(target_kernel_path):
                target_kernel_path = os.path.join(test_bench_dir, f"{kernel_name}_{source_name}")
            
            # 创建目标目录
            if not os.path.exists(target_kernel_path):
                os.makedirs(target_kernel_path)
            
            # 移动所有设计到测试基准目录
            for design_id in os.listdir(kernel_path):
                design_path = os.path.join(kernel_path, design_id)
                if os.path.isdir(design_path):
                    target_design_path = os.path.join(target_kernel_path, design_id)
                    shutil.move(design_path, target_design_path)
            
            # 如果内核目录为空，则删除它
            if not os.listdir(kernel_path):
                os.rmdir(kernel_path)
    
    # 保存选择的内核信息到日志文件
    if save_log and selected_kernels:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(test_bench_dir, f"selected_kernels_{timestamp}.txt")
        with open(log_path, 'w') as f:
            f.write(f"总共选择了 {len(selected_kernels)} 个内核作为测试集:\n\n")
            for kernel in sorted(selected_kernels):
                f.write(f"{kernel}\n")
        print(f"已将选择的内核信息保存到 {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从设计目录中选择测试基准')
    parser.add_argument('--designs_dir', type=str, required=True, help='设计文件的源目录')
    parser.add_argument('--test_bench_dir', type=str, required=True, help='测试基准的目标目录')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='选择作为测试集的内核比例（默认为0.1）')
    parser.add_argument('--no-log', action='store_true', help='不保存选择的内核信息到日志文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.designs_dir):
        print(f"错误：设计目录 '{args.designs_dir}' 不存在")
        sys.exit(1)
    
    pick_test_bench(args.designs_dir, args.test_bench_dir, args.test_ratio, not args.no_log)
    print(f"完成！测试基准已移动到 {args.test_bench_dir}")

# python pick_test_bench.py --designs_dir /home/user/zedongpeng/workspace/HLSBatchProcessor/data/designs --test_bench_dir ./test_bench --test_ratio 0.1