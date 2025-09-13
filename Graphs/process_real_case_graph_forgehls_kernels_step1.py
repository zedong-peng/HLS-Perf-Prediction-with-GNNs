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


# %%

forgehls_dataset_names = ['PolyBench', 'CHStone', 'MachSuite', "rosetta", 
                          "rtl_module", "rtl_ip", "rtl_chip", 
                          "Vitis-HLS-Introductory-Examples-flatten", 
                          "operators", "leetcode_hls_algorithms", "hls_algorithms"]
designs_base_dir = '/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels'

# 对每个数据集进行处理
for dataset_name in forgehls_dataset_names:
    print(f"\n\n========== 开始处理数据集: {dataset_name} ==========\n")
    designs_dir = f'{designs_base_dir}/{dataset_name}'

    # call dataset_csv.py to dataset_csv
    from dataset_csv import dataset_csv
    dataset_csv(designs_dir, "./csv")

    data_of_designs_json_path = f'./csv/data_of_designs_{dataset_name}.json'

    # 如果./real_case/dataset_name ./real_case/dataset_name_adb ./real_case/dataset_name _ds folder存在 那么删除后重建

    if os.path.exists(f'./real_case/{dataset_name}'):
        shutil.rmtree(f'./real_case/{dataset_name}')
    if os.path.exists(f'./real_case/{dataset_name}_adb'):
        shutil.rmtree(f'./real_case/{dataset_name}_adb')
    if os.path.exists(f'./real_case/{dataset_name}_ds'):
        shutil.rmtree(f'./real_case/{dataset_name}_ds')

    # 创建目录结构
    os.makedirs(f'./real_case/{dataset_name}')
    os.makedirs(f'./real_case/{dataset_name}_adb')
    os.makedirs(f'./real_case/{dataset_name}_ds')

    # %% [markdown]
    # The following is to extract graphs from adb files, and to save into json files.

    # %%
    def get_rtl_hash_table(root):
        """
        param: 
            root: the root of the adb file
        return:
            rtl_table: This file returns a hash table of resources and the rtlNames.
        """
        all_rtl = root.findall('*/res/*/item')
        rtl_table = {}
        if_add = False
        rep = re.compile(' \(.*\)')
        for i in all_rtl:
            res_table = {}
            rtl_name = i.find('first').text
            rtl_res = i.find('second')
            if rtl_name not in rtl_table.keys():
                for res in rtl_res.iter('item'):
                    try:
                        res_name = res.findall('first')[0].text
                        res_num = res.findall('second')[0].text
                    except BaseException:
                        # print('The RTL $',rtl_name,'& does not contain any resource info.')
                        break
                    else:
                        if res_name in res_considered:
                            res_table[res_name] = res_num
                            if_add = True
            if if_add:
                rtl_table[re.sub(rep, '', rtl_name)] = res_table
            if_add = False
        return rtl_table

    # %%
    ### parse adb files into graphs (in json)
    res_considered = ['FF', 'LUT', 'DSP']

    def parse_xml_into_graph_single(xml_file):
        prefix = ''
        G = nx.DiGraph()
        parser = et.parse(xml_file)
        root = parser.getroot()
        cdfg = root.findall('*/cdfg')[0]

        # rtl hash table
        rtl_res_table = get_rtl_hash_table(root)

        ### find edges and build the graph
        #print("Adding Edges")
        edge_id_max = -1
        for edges in cdfg.iter('edges'):
            for edge in edges.iter('item'):
                source = edge.find('source_obj').text
                sink = edge.find('sink_obj').text
                edge_id = edge.find('id').text
                edge_id_max = max(int(edge_id), edge_id_max)
                is_back_edge = edge.find('is_back_edge').text
                edge_type = edge.find('edge_type').text
                G.add_edges_from([(prefix + source, prefix + sink, {'edge_name': prefix + edge_id, 'is_back_edge': is_back_edge, 'edge_type': edge_type})])

        ### add node attributes
        #print("Adding Nodes")
        for nodes in cdfg.iter('nodes'):
            for node in nodes.findall('item'):
                node_id = node.findall('*/*/id')[0].text
                node_name = prefix + node_id
            
                if node_name not in G.nodes():
                    #print('Node %s (type: nodes) not in the graph' % node_name)
                    op_code = node.findall('opcode')[0].text
                    if op_code == 'ret':
                        G.add_node(node_name)
                        G.nodes[node_name]['node_name'] = node_name
                        G.nodes[node_name]['category']='nodes'
                        G.nodes[node_name]['bitwidth'] = node.findall('*/bitwidth')[0].text
                        G.nodes[node_name]['opcode'] = node.findall('opcode')[0].text
                        G.nodes[node_name]['m_Display'] = node.findall('m_Display')[0].text
                        G.nodes[node_name]['m_isOnCriticalPath'] = node.findall('m_isOnCriticalPath')[0].text
                        G.nodes[node_name]['m_isStartOfPath'] = node.findall('m_isStartOfPath')[0].text
                        G.nodes[node_name]['m_delay'] = node.findall('m_delay')[0].text
                        G.nodes[node_name]['m_topoIndex'] = node.findall('m_topoIndex')[0].text
                        G.nodes[node_name]['m_isLCDNode'] = node.findall('m_isLCDNode')[0].text
                        G.nodes[node_name]['m_clusterGroupNumber'] = node.findall('m_clusterGroupNumber')[0].text
                        G.nodes[node_name]['type'] = node.findall('*/*/type')[0].text
                        G.nodes[node_name]['LUT'] = '0'
                        G.nodes[node_name]['FF'] = '0'
                        G.nodes[node_name]['DSP'] = '0'
                    continue

                G.nodes[node_name]['node_name'] = node_name        
                G.nodes[node_name]['category'] = 'nodes'
                G.nodes[node_name]['bitwidth'] = node.findall('*/bitwidth')[0].text
                G.nodes[node_name]['opcode'] = node.findall('opcode')[0].text
                G.nodes[node_name]['m_Display'] = node.findall('m_Display')[0].text
                G.nodes[node_name]['m_isOnCriticalPath'] = node.findall('m_isOnCriticalPath')[0].text
                G.nodes[node_name]['m_isStartOfPath'] = node.findall('m_isStartOfPath')[0].text
                G.nodes[node_name]['m_delay'] = node.findall('m_delay')[0].text
                G.nodes[node_name]['m_topoIndex'] = node.findall('m_topoIndex')[0].text
                G.nodes[node_name]['m_isLCDNode'] = node.findall('m_isLCDNode')[0].text
                G.nodes[node_name]['m_clusterGroupNumber'] = node.findall('m_clusterGroupNumber')[0].text
                G.nodes[node_name]['type'] = node.findall('*/*/type')[0].text
                # rtl info below
                # every nodes has the three features, so we initilize them as 0.
                G.nodes[node_name]['LUT'] = '0'
                G.nodes[node_name]['FF'] = '0'
                G.nodes[node_name]['DSP'] = '0'
                t_rtlname = node.findall('*/*/rtlName')[0].text
                if t_rtlname != None:
                    # if this nodes has a rtlName info
                    if t_rtlname in rtl_res_table.keys():
                        # if this rtlName has corresponding resources info
                        # print(t_rtlname, '+++++++++++', rtl_res_table[t_rtlname])
                        res_name = rtl_res_table[t_rtlname].keys()
                        for i in res_name:
                            # rewrite the initial number with the actual number
                            G.nodes[node_name][i] = rtl_res_table[t_rtlname][i]

        ## blocks are for control signals
        for nodes in cdfg.iter('blocks'):
            for node in nodes.findall('item'):
                node_id = node.findall('*/id')[0].text
                node_name = prefix + node_id

                if node_name not in G.nodes():
                    #print('Node %s (type: blocks) not in the graph' % node_name)
                    continue
                G.nodes[node_name]['node_name'] = node_name        
                G.nodes[node_name]['category'] = 'blocks'
                G.nodes[node_name]['type'] = node.findall('*/type')[0].text
        
        ## ports are function arguments 
        for nodes in cdfg.iter('ports'):
            for node in nodes.findall('item'):
                node_id = node.findall('*/*/id')[0].text
                node_name = prefix + node_id

                if node_name not in G.nodes():
                    #print('Node %s (type: ports) not in the graph' % node_name)
                    continue
                G.nodes[node_name]['node_name'] = node_name        
                G.nodes[node_name]['category'] = 'ports'
                G.nodes[node_name]['type'] = node.findall('*/*/type')[0].text
                G.nodes[node_name]['bitwidth'] = node.findall('*/bitwidth')[0].text
                G.nodes[node_name]['direction'] = node.findall('direction')[0].text
                G.nodes[node_name]['if_type'] = node.findall('if_type')[0].text
                G.nodes[node_name]['array_size'] = node.findall('array_size')[0].text

        ## no need to keep consts as nodes in the graph
        ## remove to reduce the graph size
        for nodes in cdfg.iter('consts'):
            for node in nodes.findall('item'):
                node_id = node.findall('*/*/id')[0].text
                node_name = prefix + node_id

                if node_name not in G.nodes():
                    #print('Node %s (type: consts) not in the graph' % node_name)
                    continue
                for v in G.neighbors(node_name):
                    G.nodes[v]['const'] = node_name
                    G.nodes[v]['const-bitwidth'] = node.findall('*/bitwidth')[0].text
                # remove the const node
                G.remove_node(node_name)
                #print("const node %s removed" % node_name)

        #edge_list = list(G.edges)
        #print(edge_list)
        #node_list = list(G.nodes)
        #print(node_list)
        return G

    # %%
    ### save one graph into json
    def json_save(G, fname):
        f = open(fname + '.json', 'w')
        G_dict = dict(nodes=[[n, G.nodes[n]] for n in G.nodes()], \
                    edges=[(e[0], e[1], G.edges[e]) for e in G.edges()])
        json.dump(G_dict, f)
        f.close()

    # %%
    ### save the graphs into json
    def json_save_graphs(Gs, fname):
        f = open(fname + '.json', 'w')
        G_dict = dict()
        G_dict['nodes'] = []
        G_dict['edges'] = []
        for G in Gs:
            for n in G.nodes():
                G_dict['nodes'].append([n, G.nodes[n]])
            for e in G.edges():
                G_dict['edges'].append((e[0], e[1], G.edges[e]))
        json.dump(G_dict, f)
        f.close()

    # %%
    ### read the actual resource
    def get_real_perf(fname):
        f = open(fname + '.json', 'r')
        d = json.load(f)
        f.close()
        DSP=d['DSP']
        LUT=d['LUT']
        FF=d['FF']

        return DSP, LUT, FF

    # %%
    ### prepare adb files from cpp/c files to _adb folder
    import os
    import shutil
    import tqdm
    import re
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing

    def find_cpp_c_files(search_dir, max_depth=5):
        """
        在指定目录及其子目录（最大深度 max_depth）中搜索 .cpp 和 .c 文件。
        
        :param search_dir: 要搜索的目录
        :param max_depth: 最大搜索深度（默认值为5）
        :return: 包含 .cpp 和 .c 文件路径的列表
        """
        cpp_c_files = []
        
        # 使用列表推导式快速收集所有目录
        all_dirs = []
        for root, dirs, _ in os.walk(search_dir):
            # 过滤 .autopilot 目录
            if '.autopilot' in dirs:
                dirs.remove('.autopilot')
            # 计算当前目录的深度
            rel_path = os.path.relpath(root, search_dir)
            depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
            
            if depth < max_depth:
                all_dirs.append(root)
        
        # 使用tqdm显示进度
        for root in tqdm.tqdm(all_dirs, desc="搜索目录"):
            # 直接扩展列表而不是逐个添加
            cpp_c_files.extend([os.path.join(root, file) for file in os.listdir(root) 
                            if file.endswith(('.cpp', '.c'))])
        
        return cpp_c_files

    def process_single_file(c_file, designs_dir):
        """处理单个C/C++文件并生成对应的.adb文件"""
        try:
            # 获取绝对路径
            abs_path = os.path.abspath(c_file)
            # 解析路径获取source_name, kernel_name（kernels目录中没有design_id层级）
            path_parts = abs_path.split(os.sep)
            if len(path_parts) < 3:
                return []
                
            file_name = os.path.splitext(path_parts[-1])[0]
            kernel_name = path_parts[-2]  # 现在kernel_name是倒数第二个
            source_name = path_parts[-3]  # source_name是倒数第三个
            design_id = "unknown"  # 设置为unknown，因为没有design_id层级
            
            # 构建新的文件名前缀
            prefix = f"{source_name}-{kernel_name}-{design_id}-"
            
            # 查找对应的.adb文件
            adb_dir = os.path.dirname(c_file)
            
            # 在c_file所在路径的./**多级子目录找 .adb文件 注意最多一个后缀
            adb_files = []
            for root, dirs, files in os.walk(adb_dir):
                for file in files:
                    if file.endswith('.adb') and len(file.split('.')) == 2:
                        adb_files.append(os.path.join(root, file))
            
            # 去除重复项
            adb_files = list(set(adb_files))

            # 创建目标目录（提前创建以避免多线程冲突）
            target_dir = os.path.join('real_case', source_name + '_adb')
            os.makedirs(target_dir, exist_ok=True)
            
            # 批量复制文件
            for adb_file in adb_files:
                original_adb_name = os.path.basename(adb_file)
                new_adb_name = f"{prefix}{original_adb_name}"
                target_path = os.path.join(target_dir, new_adb_name)
                shutil.copy2(adb_file, target_path)
                
            return []
        except Exception as e:
            return [f"处理 {c_file} 时出错: {str(e)}"]

    def process_c_files_to_adb(designs_dir, c_files):
        """
        处理C/C++文件并生成对应的.adb文件，使用多进程和多线程加速
        
        Args:
            designs_dir: 设计文件根目录
            c_files: C/C++文件列表
        """
        # 创建所有可能需要的目标目录，避免线程冲突
        os.makedirs('real_case', exist_ok=True)
        
        # 计算最佳工作进程数
        cpu_count = multiprocessing.cpu_count()
        workers = min(cpu_count * 8, 256)  # 使用更多的工作线程
        
        # 创建进度条
        pbar = tqdm.tqdm(total=len(c_files), desc="处理文件")
        
        # 使用线程池加速处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_single_file, c_file, designs_dir) for c_file in c_files]
            
            # 处理结果
            for future in as_completed(futures):
                try:
                    results = future.result()
                    # 只在有错误时输出
                    if results:
                        print(f"处理文件出现 {len(results)} 个错误")
                except Exception as e:
                    print(f"处理文件时发生异常: {str(e)}")
                finally:
                    pbar.update(1)
        
        pbar.close()

    # 主执行代码
    print("开始搜索C/C++文件...")
    c_files = find_cpp_c_files(designs_dir)
    print(f"找到 {len(c_files)} 个C/C++文件，开始处理...")

    # 统计设计数量
    design_count = {}
    for c_file in c_files:
        try:
            path_parts = os.path.abspath(c_file).split(os.sep)
            if len(path_parts) >= 3:
                source_name = path_parts[-3]
                kernel_name = path_parts[-2]
                key = f"{source_name}-{kernel_name}"
                if key not in design_count:
                    design_count[key] = set()
                design_count[key].add("unknown")  # 因为没有design_id层级
        except Exception as e:
            print(f"统计设计时出错: {str(e)}")

    # 打印统计结果
    print("\n设计数量统计:")
    total_designs = 0
    for key, designs in design_count.items():
        design_num = len(designs)
        total_designs += design_num
        print(f"  {key}: {design_num}个设计")
    print(f"总计: {total_designs}个设计\n")

    # 批量处理以提高效率，增加批次大小
    batch_size = 5000  # 增加批次大小
    total_batches = (len(c_files) + batch_size - 1) // batch_size

    source_names = set()
    for c_file in c_files:
        try:
            path_parts = os.path.abspath(c_file).split(os.sep)
            if len(path_parts) >= 3:
                source_names.add(path_parts[-3])
        except:
            pass

    for source_name in source_names:
        os.makedirs(os.path.join('real_case', source_name + '_adb'), exist_ok=True)

    # 处理批次
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(c_files))
        batch = c_files[start_idx:end_idx]
        print(f"处理批次 {i+1}/{total_batches}，文件数量: {len(batch)}")
        process_c_files_to_adb(designs_dir, batch)

    print("处理完成！")


    # %%
    ### generate posthls_*.json post hls data
    df = pd.read_json(data_of_designs_json_path, orient='records')
    for index, row in df.iterrows():
        source_name = row['source_name']
        kernel_name = row['algo_name']
        design_id = row['design_id']
        dsp = row['DSP']
        lut = row['LUT']
        ff = row['FF']
        save_path = os.path.join('real_case', f'{dataset_name}', f'posthls_{source_name}-{kernel_name}-{design_id}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({"DSP": dsp, "LUT": lut, "FF": ff}, f)
        print(f"Saved {save_path}")
        print(index)



    # %%
    result_dir = f'{dataset_name}'
    graph_dir = f'{dataset_name}_adb/'

    # %%
    ### get subgraphs in one application
    graph_mapping = dict()
    for adb_file in glob.glob('real_case/' + graph_dir + '*.adb'):
        _, _, file_name = adb_file.split('/')
        # 获取最后一个-之前和之后的部分
        parts = file_name.rsplit('-', 1)
        fname = parts[0]  # 最后一个-之前的所有内容
        func_name = parts[1]  # 最后一个-之后的内容
        if fname not in graph_mapping:
            graph_mapping[fname] = [func_name]
        else:
            graph_mapping[fname].append(func_name)

    # %%
    def check_max_node_id(node_string):
        node_array=[]
        for n in node_string:
            node_array.append(int(n))
        max_id=max(node_array)
        return max_id

    # %%
    ### 最终阶段：将图保存为json文件
    os.makedirs(os.path.join('real_case', result_dir), exist_ok=True)

    # 使用多进程并行处理
    from multiprocessing import Pool
    from functools import partial
    import tqdm

    def process_single_file(adb_file, max_id=0):
        try:
            g = parse_xml_into_graph_single(adb_file)
            if max_id > 0:
                # relabel nodes
                mapping = {n:str(int(n)+max_id) for n in g.nodes}
                g = nx.relabel_nodes(g, mapping)
            return g
        except Exception as e:
            print(f"处理文件 {adb_file} 时出错: {str(e)}")
            return None

    def process_fname(fname, graph_dir, result_dir):
        try:
            graph_num = len(graph_mapping[fname])
            print(f"Processing {fname} with {graph_num} graphs")
            # 预先获取所有匹配的文件列表
            adb_files = glob.glob('real_case/' + graph_dir + fname + '-*')
            
            if graph_num > 1:
                max_id = 0
                G = []
                for adb_file in adb_files:
                    g = process_single_file(adb_file, max_id)
                    if g is not None:
                        G.append(g)
                        max_id = check_max_node_id(g.nodes) + 1
                if G:
                    json_save_graphs(G, 'real_case/' + result_dir + '/' + fname)
            else:
                if adb_files:
                    g = process_single_file(adb_files[0])
                    if g is not None:
                        json_save(g, 'real_case/' + result_dir + '/' + fname)
            return fname
        except Exception as e:
            print(f"处理 {fname} 时出错: {str(e)}")
            return None

    # 使用进程池并行处理
    total_files = len(graph_mapping.keys())
    print(f"开始处理总共 {total_files} 个文件...")

    with Pool() as pool:
        process_func = partial(process_fname, graph_dir=graph_dir, result_dir=result_dir)
        results = []
        for result in tqdm.tqdm(pool.imap_unordered(process_func, list(graph_mapping.keys())), total=total_files):
            results.append(result)

    # 统计成功和失败的文件数
    successful = [r for r in results if r is not None]
    print(f"处理完成! 成功: {len(successful)}/{total_files}")


    # %% [markdown]
    # The following is to process graphs into dataset format. 

    # %%
    ### features for numerical rtl resource

    allowable_features = {
        'node_category' : ['nodes', 'blocks', 'ports', 'misc'], 
        'bitwidth' : list(range(0, 256)) + ['misc'], 
        'opcode_category' : ['terminator','binary_unary', 'bitwise', 'conversion','memory','aggregate','other','misc'], 
        'possible_opcode_list' : [
            'br', 'ret', 'switch',
            'add', 'dadd', 'fadd', 'sub', 'dsub', 'fsub', 'mul', 'dmul', 'fmul', 'udiv', 'ddiv', 'fdiv', 'sdiv', 'urem', 'srem', 'frem', 'dexp', 'dsqrt',
            'shl', 'lshr', 'ashr', 'and', 'xor', 'or',
            'uitofp', 'sitofp', 'uitodp', 'sitodp', 'bitconcatenate', 'bitcast', 'zext', 'sext', 'fpext', 'trunc', 'fptrunc',
            'extractvalue', 'insertvalue',
            'alloca', 'load', 'store', 'read', 'write', 'getelementptr',
            'phi', 'call', 'icmp', 'dcmp', 'fcmp', 'select', 'bitselect', 'partselect', 'mux', 'dacc',
            'misc'
        ],
        'possible_is_start_of_path': [0, 1, 'misc'],
        'possible_is_LCDnode':[0, 1, 'misc'],
        'possible_cluster_group_num': [-1] + list(range(0, 256)) + ['misc'],
        'LUT': list(range(0, 1000)) + ['misc'],
        'DSP': list(range(0, 11)) + ['misc'],
        'FF': list(range(0, 1000)) + ['misc'],
        
        'possible_edge_type_list' : [1, 2, 3, 'misc'], 
        'possible_is_back_edge': [0, 1],
    }

    def safe_index(l, e):
        """
        Return index of element e in list l. If e is not present, return the last index
        """
        try:
            return l.index(e)
        except:
            return len(l) - 1

    def opcode_type(opcode):
        t = 'misc'
        if opcode in {'br', 'ret', 'switch'}:
            t = 'terminator'
        elif opcode in {'add', 'dadd', 'fadd', 'sub', 'dsub', 'fsub', 'mul', 'dmul', 'fmul', 'udiv', 'ddiv', 'fdiv', 'sdiv', 'urem', 'srem', 'frem', 'dexp', 'dsqrt'}:
            t = 'binary_unary'
        elif opcode in {'shl', 'lshr', 'ashr', 'and', 'xor', 'or'}:
            t = 'bitwise'
        elif opcode in {'uitofp', 'sitofp', 'uitodp', 'sitodp', 'bitconcatenate', 'bitcast', 'zext', 'sext', 'fpext', 'trunc', 'fptrunc'}:
            t = 'conversion'
        elif opcode in {'alloca', 'load', 'store', 'read', 'write', 'getelementptr'}:
            t = 'memory'
        elif opcode in {'extractvalue', 'insertvalue'}:
            t = 'aggregate'
        elif opcode in {'phi', 'call', 'icmp', 'dcmp', 'fcmp', 'select', 'bitselect', 'partselect', 'mux', 'dacc', 'sparsemux'}:
            t = 'other'
        if t == 'misc':
            print(f"opcode: {opcode}")
        return t



    def node_to_feature_vector(node):
        """
        Converts node object to feature list of indices
        :return: list
        """

        if node=={}:
            node_feature = [
                    len(allowable_features['node_category'])-1,
                    len(allowable_features['bitwidth'])-1,
                    len(allowable_features['opcode_category'])-1,
                    len(allowable_features['possible_opcode_list'])-1,
                    len(allowable_features['possible_is_start_of_path'])-1,
                    len(allowable_features['possible_is_LCDnode'])-1,
                    len(allowable_features['possible_cluster_group_num'])-1,
                    len(allowable_features['LUT'])-1,
                    len(allowable_features['DSP'])-1,
                    len(allowable_features['FF'])-1
                    ]
            return node_feature
            
        if node['category']=='nodes':
            node_feature = [
                    safe_index(allowable_features['node_category'], node['category']),
                    safe_index(allowable_features['bitwidth'], int(node['bitwidth'])),
                    safe_index(allowable_features['opcode_category'], opcode_type(node['opcode'])),
                    safe_index(allowable_features['possible_opcode_list'], node['opcode']),
                    safe_index(allowable_features['possible_is_start_of_path'], int(node['m_isStartOfPath'])),
                    safe_index(allowable_features['possible_is_LCDnode'], int(node['m_isLCDNode'])),
                    safe_index(allowable_features['possible_cluster_group_num'], int(node['m_clusterGroupNumber'])),
                    safe_index(allowable_features['LUT'], int(node['LUT'])),
                    safe_index(allowable_features['DSP'], int(node['DSP'])),
                    safe_index(allowable_features['FF'], int(node['FF']))
                    ]
        elif node['category']=='ports':
            node_feature = [
                    safe_index(allowable_features['node_category'], node['category']),
                    safe_index(allowable_features['bitwidth'], int(node['bitwidth'])),
                    len(allowable_features['opcode_category'])-1,
                    len(allowable_features['possible_opcode_list'])-1,
                    len(allowable_features['possible_is_start_of_path'])-1,
                    len(allowable_features['possible_is_LCDnode'])-1,
                    len(allowable_features['possible_cluster_group_num'])-1,
                    len(allowable_features['LUT'])-1,
                    len(allowable_features['DSP'])-1,
                    len(allowable_features['FF'])-1
                    ]
        elif node['category']=='blocks':
            node_feature = [
                    safe_index(allowable_features['node_category'], node['category']),
                    len(allowable_features['bitwidth'])-1,
                    len(allowable_features['opcode_category'])-1,
                    len(allowable_features['possible_opcode_list'])-1,
                    len(allowable_features['possible_is_start_of_path'])-1,
                    len(allowable_features['possible_is_LCDnode'])-1,
                    len(allowable_features['possible_cluster_group_num'])-1,
                    len(allowable_features['LUT'])-1,
                    len(allowable_features['DSP'])-1,
                    len(allowable_features['FF'])-1
                    ]
        return node_feature

    def get_node_feature_dims():
        return list(map(len, [
            allowable_features['node_category'],
            allowable_features['bitwidth'],
            allowable_features['opcode_category'],
            allowable_features['possible_opcode_list'],
            allowable_features['possible_is_start_of_path'],
            allowable_features['possible_is_LCDnode'],
            allowable_features['possible_cluster_group_num'],
            allowable_features['LUT'],
            allowable_features['DSP'],
            allowable_features['FF'],
            ]))


    def edge_to_feature_vector(edge):
        """
        Converts edge to feature list of indices
        :return: list
        """
        bond_feature = [
                    safe_index(allowable_features['possible_edge_type_list'], int(edge['edge_type'])),
                    allowable_features['possible_is_back_edge'].index(int(edge['is_back_edge']))
                ]
        return bond_feature

    def get_edge_feature_dims():
        return list(map(len, [
            allowable_features['possible_edge_type_list'],
            allowable_features['possible_is_back_edge']
            ]))


    # %%
    #result_dir='PolyBench/'
    #prefix='polybench_'
    # result_dir='CHStone/'
    # prefix='chstone_'
    result_dir=f'{dataset_name}/'
    prefix='posthls_'


    # %%
    ### graphs in json transformed into csv format
    graph_mapping_list = []
    num_node_list = []
    num_edge_list = []

    DSP = []
    LUT = []
    FF = []

    node_feat = []
    edge_list = []
    edge_feat = []

    processed_count = 0
    skipped_count = 0
    total_files = len(glob.glob('real_case/' + result_dir + prefix + '*.json'))
    print(f"总文件数: {total_files}")

    from tqdm import tqdm
    for perf_file in tqdm(glob.glob('real_case/' + result_dir + prefix + '*.json'), desc="处理文件"):
        _, _, file_name = perf_file.split('/')
        graph_name = file_name.replace(prefix,'')

        try:
            # print(f'real_case/'+result_dir + graph_name)
            f = open('real_case/'+result_dir + graph_name, 'r')
            d = json.load(f)
            f.close()
            nodes=d['nodes']
            edges=d['edges']

            try:
                dsp, lut, ff = get_real_perf(perf_file.replace('.json',''))
                
                num_node_list.append(len(nodes))
                num_edge_list.append(len(edges))
                graph_mapping_list.append(result_dir + graph_name)
                
                DSP.append(dsp)
                LUT.append(lut)
                FF.append(ff)

                node_index_map = dict() # map the node name to the index
                index = 0

                for n in nodes:
                    if n[0] not in node_index_map:
                        node_index_map[n[0]] = index
                    node_feat.append(node_to_feature_vector(n[1]))
                    index = index + 1
                
                for e in edges:
                    source = node_index_map[e[0]]
                    sink = node_index_map[e[1]]
                    edge_list.append([source,sink])
                    edge_feat.append(edge_to_feature_vector(e[2]))
                processed_count += 1
            except Exception as e:
                print(f"Error processing {perf_file}: {e}") 
                # 打印完整的报错
                print(traceback.format_exc())

                skipped_count += 1
                continue
        except Exception as e:
            skipped_count += 1
            continue

    print(f"处理完成: {processed_count}/{total_files} 文件")
    print(f"跳过: {skipped_count}/{total_files} 文件 ({skipped_count/total_files*100:.2f}%)")

    # %%
    ### save graphs into csv files

    ds_dir = f'{dataset_name}_ds'                
    save_dir = 'real_case/' + ds_dir + '/' # the directory to save real cases, three benchmarks are saved separately in this stage
    os.makedirs(save_dir, exist_ok=True)
    mapping = pd.DataFrame({'orignal code':graph_mapping_list , 'DSP' : DSP , 'LUT' : LUT, 'FF' : FF})
    NODE_num = pd.DataFrame(num_node_list) # number of nodes in each graph 
    NODE = pd.DataFrame(node_feat) # node features
    EDGE_num = pd.DataFrame(num_edge_list) # number of edges in each graph
    EDGE_list = pd.DataFrame(edge_list) # edge (source, end)
    EDGE_feat = pd.DataFrame(edge_feat) # edge features

    graph_label_dsp = pd.DataFrame(DSP)
    graph_label_lut = pd.DataFrame(LUT)
    graph_label_ff = pd.DataFrame(FF)

    # save into csv files
    mapping.to_csv(save_dir + 'mapping.csv', index = False)
    NODE_num.to_csv(save_dir + 'num-node-list.csv', index = False, header = False)
    NODE.to_csv(save_dir + 'node-feat.csv', index = False, header = False)

    EDGE_num.to_csv(save_dir + 'num-edge-list.csv', index = False, header = False)
    EDGE_list.to_csv(save_dir + 'edge.csv', index = False, header=False)
    EDGE_feat.to_csv(save_dir + 'edge-feat.csv', index = False, header = False)

    graph_label_dsp.to_csv(save_dir + 'graph-label-dsp.csv', index = False, header = False)
    graph_label_lut.to_csv(save_dir + 'graph-label-lut.csv', index = False, header = False)
    graph_label_ff.to_csv(save_dir + 'graph-label-ff.csv', index = False, header = False)

    print(f"The step1 process is done. Please run python process_real_case_graph_forgehls_step2.py to merge all real case.")