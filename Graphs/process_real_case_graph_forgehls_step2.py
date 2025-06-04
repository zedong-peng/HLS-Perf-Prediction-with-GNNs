# %%
import xml.etree.cElementTree as et
import networkx as nx
import matplotlib.pyplot as plt
import json
import subprocess
import time
from os import path
import glob
import pandas as pd
import re
import os
import shutil
import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# 添加dataset_name变量定义
dataset_name = "forgehls"


# %% [markdown]
# The following is to merge real-case benchmarks with synthetic cdfg.

# %%
### merge all real case
# 
forgehls_dataset_names = ['PolyBench', 'CHStone', 'MachSuite', "rosetta", 
                          "rtl_module", "rtl_ip", "rtl_chip", 
                          "Vitis-HLS-Introductory-Examples-flatten", 
                          "operators", "leetcode_hls_algorithms", "hls_algorithms"]

case_dir_all = []
for dataset_name in forgehls_dataset_names:
    ds_dir = f'real_case/{dataset_name}_ds/'
    if os.path.exists(ds_dir):
        case_dir_all.append(dataset_name)
        print(f"找到数据集: {dataset_name}")
    else:
        print(f"数据集目录不存在: {ds_dir}")

print(f"\n总共找到 {len(case_dir_all)} 个可用数据集:")
for dir_path in case_dir_all:
    print(f" - {dir_path}")

if not case_dir_all:
    print(f"警告: 未找到匹配的目录")

graph_mapping_list = []
edge_feat = []
edge_list = []
node_feat = []

DSP = []
LUT = []
FF = []

num_node_list = []
num_edge_list = []


for case_dir in case_dir_all:
    graph_mapping_list += pd.read_csv(case_dir + 'mapping.csv').values.tolist()
    edge_feat += pd.read_csv(case_dir + 'edge-feat.csv', header = None).values.tolist()
    edge_list += pd.read_csv(case_dir + 'edge.csv', header = None).values.tolist()
    node_feat += pd.read_csv(case_dir + 'node-feat.csv', header = None).values.tolist()

    DSP += pd.read_csv(case_dir + 'graph-label-dsp.csv', header = None).values.tolist()
    LUT += pd.read_csv(case_dir + 'graph-label-lut.csv', header = None).values.tolist()
    FF += pd.read_csv(case_dir + 'graph-label-ff.csv', header = None).values.tolist()

    num_node_list += pd.read_csv(case_dir + 'num-node-list.csv', header = None).values.tolist()
    num_edge_list += pd.read_csv(case_dir + 'num-edge-list.csv', header = None).values.tolist()

# %%
### save merged dataset

save_dir = f'real_case/dataset_ready_for_GNN_csv/{dataset_name}/'
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

mapping = pd.DataFrame({'orignal code' : graph_mapping_list, 'DSP' : DSP, 'LUT' : LUT, 'FF' : FF})
NODE_num = pd.DataFrame(num_node_list)
EDGE_num = pd.DataFrame(num_edge_list)

graph_label_dsp = pd.DataFrame(DSP)
graph_label_lut = pd.DataFrame(LUT)
graph_label_ff = pd.DataFrame(FF)

NODE = pd.DataFrame(node_feat)
EDGE_list = pd.DataFrame(edge_list)
EDGE_feat = pd.DataFrame(edge_feat)


mapping.to_csv(save_dir + 'mapping.csv', index = False)
NODE_num.to_csv(save_dir + 'num-node-list.csv', index = False, header = False)
EDGE_num.to_csv(save_dir + 'num-edge-list.csv', index = False, header = False)

graph_label_dsp.to_csv(save_dir + 'graph-label-dsp.csv', index = False, header = False)
graph_label_lut.to_csv(save_dir + 'graph-label-lut.csv', index = False, header = False)
graph_label_ff.to_csv(save_dir + 'graph-label-ff.csv', index = False, header = False)

NODE.to_csv(save_dir + 'node-feat.csv', index = False, header = False)
EDGE_list.to_csv(save_dir + 'edge.csv', index = False, header = False)
EDGE_feat.to_csv(save_dir + 'edge-feat.csv', index = False, header = False)

# %% [markdown]
# The following is to generate training/valid/test set.

# %%
### for training set
from sklearn import model_selection

basis = len(pd.read_csv(f'{save_dir}/graph-label-dsp.csv', header=None))
print(basis)

# 将基础数据集分割为训练集(80%)、验证集(10%)和测试集(10%)
indices = [i for i in range(basis)]
train_indices, temp_indices = model_selection.train_test_split(indices, train_size=0.8, random_state=42)
valid_indices, test_indices = model_selection.train_test_split(temp_indices, train_size=0.5, random_state=42)

# 保存训练集
train_list = pd.DataFrame(sorted(train_indices))
train_list.to_csv(save_dir + 'train.csv', index=False, header=False)

# 保存验证集
valid_list = pd.DataFrame(sorted(valid_indices))
valid_list.to_csv(save_dir + 'valid.csv', index=False, header=False)

# 保存测试集
test_list = pd.DataFrame(sorted(test_indices))
test_list.to_csv(save_dir + 'test.csv', index=False, header=False)

# %%
# 按照以下结构组织数据集文件:
# cdfg_{metric}_all_numerical_{dataset_name}/
#   mapping/
#     mapping.csv.gz
#   split/scaffold/
#     test.csv.gz
#     train.csv.gz
#     valid.csv.gz
#   raw/
#     node-feat.csv.gz
#     edge.csv.gz
#     edge-feat.csv.gz
#     graph-label-{metric}.csv.gz
#     num-node-list.csv.gz
#     num-edge-list.csv.gz

import os
import gzip
import shutil
import concurrent.futures
import multiprocessing


# 创建数据集目录结构并压缩文件
def create_dataset_structure(metric, dataset_name):
    dataset_dir = f"./dataset/cdfg_{metric}_all_numerical_{dataset_name}/"

    os.makedirs(f"{dataset_dir}mapping", exist_ok=False)
    os.makedirs(f"{dataset_dir}split/scaffold", exist_ok=False)
    os.makedirs(f"{dataset_dir}raw", exist_ok=False)
    
    # 定义压缩文件的函数
    def compress_file(src_path, dst_path):
        with open(src_path, 'rb') as f_in:
            with gzip.open(dst_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return dst_path
    
    # 准备需要压缩的文件列表
    compression_tasks = [
        # mapping文件
        (f"{save_dir}mapping.csv", f"{dataset_dir}mapping/mapping.csv.gz"),
        
        # split文件
        (f"{save_dir}train.csv", f"{dataset_dir}split/scaffold/train.csv.gz"),
        (f"{save_dir}valid.csv", f"{dataset_dir}split/scaffold/valid.csv.gz"),
        (f"{save_dir}test.csv", f"{dataset_dir}split/scaffold/test.csv.gz"),
        
        # raw文件
        (f"{save_dir}node-feat.csv", f"{dataset_dir}raw/node-feat.csv.gz"),
        (f"{save_dir}edge.csv", f"{dataset_dir}raw/edge.csv.gz"),
        (f"{save_dir}edge-feat.csv", f"{dataset_dir}raw/edge-feat.csv.gz"),
        (f"{save_dir}num-node-list.csv", f"{dataset_dir}raw/num-node-list.csv.gz"),
        (f"{save_dir}num-edge-list.csv", f"{dataset_dir}raw/num-edge-list.csv.gz"),
        (f"{save_dir}graph-label-{metric}.csv", f"{dataset_dir}raw/graph-label.csv.gz")
    ]
    
    # 使用线程池并行压缩文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(compress_file, src, dst) for src, dst in compression_tasks]
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            try:
                completed_file = future.result()
                print(f"已完成: {completed_file}")
            except Exception as e:
                print(f"压缩文件时出错: {str(e)}")
    
    print(f"数据集 cdfg_{metric}_all_numerical_{dataset_name} 创建完成")

# 使用进程池并行创建三个指标的数据集
metrics = ["lut", "ff", "dsp"]
for metric in metrics:
    dataset_dir = f"./dataset/cdfg_{metric}_all_numerical_{dataset_name}/"
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(create_dataset_structure, metric, dataset_name) for metric in metrics]
    
    # 等待所有数据集创建完成
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"创建数据集时出错: {str(e)}")

print("所有数据集创建完成")
print("Please check the dataset in the ./dataset/")



