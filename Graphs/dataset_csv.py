import os
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import concurrent.futures
import multiprocessing

def gather_csynth_data(root_dir, output_csv):
    print("Gathering csynth data...")
    # 将路径转换为 Path 对象，便于处理
    root_dir = Path(root_dir)
    output_csv = Path(output_csv)

    # 确保输出文件的父目录存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # 首先收集所有csynth.xml文件路径
    csynth_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file == 'csynth.xml':
                file_path = os.path.join(root, file)
                csynth_files.append(file_path)
    
    # 并行处理文件
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for result in executor.map(process_csynth_file, csynth_files):
            if result:  # 如果结果不是None，则添加到结果列表
                results.append(result)
    
    # 写入CSV文件
    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入CSV文件的表头
        csv_writer.writerow(['File Path', 'Part', 'TargetClockPeriod', 'Best-caseLatency', 'Worst-caseLatency', 'BRAM_18K', 'LUT', 'DSP', 'FF', 'Avialable_BRAM_18K', 'Avialable_LUT', 'Avialable_DSP', 'Avialable_FF'])
        
        # 写入所有结果
        csv_writer.writerows(results)
    
    print(f"Gathering raw data done. Saved in {output_csv}")

def process_csynth_file(file_path):
    try:
        # 解析 XML 文件
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 提取所需信息
        part = root.find('.//UserAssignments/Part').text
        target_clock_period = root.find('.//UserAssignments/TargetClockPeriod').text
        best_case_latency = root.find('.//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency').text
        worst_case_latency = root.find('.//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency').text
        bram_18k = root.find('.//AreaEstimates/Resources/BRAM_18K').text
        lut = root.find('.//AreaEstimates/Resources/LUT').text
        dsp = root.find('.//AreaEstimates/Resources/DSP').text
        ff = root.find('.//AreaEstimates/Resources/FF').text
        available_bram_18k = root.find('.//AreaEstimates/AvailableResources/BRAM_18K').text
        available_lut = root.find('.//AreaEstimates/AvailableResources/LUT').text
        available_dsp = root.find('.//AreaEstimates/AvailableResources/DSP').text
        available_ff = root.find('.//AreaEstimates/AvailableResources/FF').text

        # 返回一行数据
        return [file_path, part, target_clock_period, best_case_latency, worst_case_latency, 
                bram_18k, lut, dsp, ff, available_bram_18k, available_lut, available_dsp, available_ff]
    except ET.ParseError as e:
        print(f"XML 解析错误: {file_path}, 错误信息: {str(e)}")
    except AttributeError as e:
        print(f"缺少必要的 XML 元素: {file_path}, 错误信息: {str(e)}")
    return None

def print_info(df):
    print(f"kernels number: {df['algo_name'].nunique() if 'algo_name' in df.columns else 0} designs number: {df.shape[0]}")

def base_feature(df):
    if df.empty:
        print("DataFrame为空，无法添加特征")
        return df
        
    # ResourceMetric - 使用向量化操作代替apply
    df['BRAM_18K'] = pd.to_numeric(df['BRAM_18K'], errors='coerce')
    df['LUT'] = pd.to_numeric(df['LUT'], errors='coerce')
    df['DSP'] = pd.to_numeric(df['DSP'], errors='coerce')
    df['FF'] = pd.to_numeric(df['FF'], errors='coerce')
    df['Avialable_BRAM_18K'] = pd.to_numeric(df['Avialable_BRAM_18K'], errors='coerce')
    df['Avialable_LUT'] = pd.to_numeric(df['Avialable_LUT'], errors='coerce')
    df['Avialable_DSP'] = pd.to_numeric(df['Avialable_DSP'], errors='coerce')
    df['Avialable_FF'] = pd.to_numeric(df['Avialable_FF'], errors='coerce')
    
    bram_ratio = df['BRAM_18K'] / df['Avialable_BRAM_18K'].replace(0, float('inf'))
    lut_ratio = df['LUT'] / df['Avialable_LUT'].replace(0, float('inf'))
    dsp_ratio = df['DSP'] / df['Avialable_DSP'].replace(0, float('inf'))
    ff_ratio = df['FF'] / df['Avialable_FF'].replace(0, float('inf'))
    
    # 处理无穷大值
    bram_ratio = bram_ratio.replace(float('inf'), 0)
    lut_ratio = lut_ratio.replace(float('inf'), 0)
    dsp_ratio = dsp_ratio.replace(float('inf'), 0)
    ff_ratio = ff_ratio.replace(float('inf'), 0)
    
    df['ResourceMetric'] = (bram_ratio + lut_ratio + dsp_ratio + ff_ratio) / 4

    # 提取路径信息 - 使用pandas的str方法代替apply
    df['File Path'] = df['File Path'].astype(str)
    path_parts = df['File Path'].str.split('/')
    
    # 计算每个路径的长度，以确保我们不会尝试访问不存在的索引
    max_parts = path_parts.str.len().max()
    
    # 只有当路径足够长时才提取相应的部分
    if max_parts >= 6:
        df['design_id'] = path_parts.str[-6]
    else:
        df['design_id'] = 'unknown'
        
    if max_parts >= 7:
        df['algo_name'] = path_parts.str[-7]
    else:
        df['algo_name'] = 'unknown'
        
    if max_parts >= 8:
        df['source_name'] = path_parts.str[-8]
    else:
        df['source_name'] = 'unknown'

    df = df.reset_index(drop=True)
    print_info(df)
    return df

def delete_undef(df):
    if df.empty:
        print("DataFrame为空，无法处理undefined值")
        return df
        
    # 将 'undef' 转换为 NaN
    df['Best-caseLatency'] = df['Best-caseLatency'].replace('undef', pd.NA)
    
    # 按 'algo_name' 分组过滤
    valid_groups = []
    for algo_name, group in df.groupby('algo_name'):
        if not group['Best-caseLatency'].isna().any():
            valid_groups.append(group)
    
    # 如果没有有效的组，返回空DataFrame
    if not valid_groups:
        print("没有找到有效的设计（所有设计的Best-caseLatency都包含undef）")
        return pd.DataFrame(columns=df.columns)
    
    df = pd.concat(valid_groups)
    df = df.reset_index(drop=True)
    
    # 转换为数值类型
    df['Best-caseLatency'] = pd.to_numeric(df['Best-caseLatency'], errors='coerce')
    df['Worst-caseLatency'] = pd.to_numeric(df['Worst-caseLatency'], errors='coerce')

    print(f"After deleting 'Best-caseLatency' = 'undef' cases")
    print_info(df)

    # 优化：一次性处理所有算法的零延迟情况
    grouped = df.groupby('algo_name')
    
    # 对每个组进行处理
    results = []
    for algo_name, group in grouped:
        # 对每个组内的数据进行排序
        group = group.sort_values(by=['Best-caseLatency', 'ResourceMetric'])
        group = group.reset_index(drop=True)
        
        # 区分零延迟和非零延迟组
        non_zero_latency_group = group[group['Best-caseLatency'] != 0]
        
        if len(non_zero_latency_group) > 0:
            results.append(non_zero_latency_group)
        else:
            # 如果没有非零延迟组，添加整个组
            results.append(group)
    
    # 合并结果
    if results:
        df = pd.concat(results, ignore_index=True)
    else:
        df = pd.DataFrame(columns=df.columns)
    
    print(f"After deleting zero latency cases")
    print_info(df)
    
    return df

def delete_overlap_and_overfitting(df):
    if df.empty:
        print("DataFrame为空，无法处理重叠和过拟合")
        return df
        
    # 去除重复的 'Best-caseLatency' 和 'ResourceMetric' 组合
    df = df.drop_duplicates(subset=['algo_name', 'Best-caseLatency', 'ResourceMetric'], keep='first')

    print(f"After deleting duplicate 'Best-caseLatency' and 'ResourceMetric' combinations")
    print_info(df)

    # 改进后的重叠情况删除算法
    def remove_invalid_designs_optimized(group):
        # 按资源和延迟排序
        group = group.sort_values(by=['ResourceMetric', 'Best-caseLatency'])
        
        # 创建一个布尔掩码来标识要保留的行
        keep_mask = pd.Series(True, index=group.index)
        
        # 两两比较以找出重叠设计
        for i in range(len(group)):
            if not keep_mask.iloc[i]:
                continue  # 如果该设计已标记为删除，则跳过
                
            resource_i = group['ResourceMetric'].iloc[i]
            latency_i = group['Best-caseLatency'].iloc[i]
            
            # 遍历后续设计
            for j in range(i+1, len(group)):
                if not keep_mask.iloc[j]:
                    continue  # 如果该设计已标记为删除，则跳过
                    
                resource_j = group['ResourceMetric'].iloc[j]
                latency_j = group['Best-caseLatency'].iloc[j]
                
                # 检查是否重叠
                if resource_i == resource_j and latency_i == latency_j:
                    keep_mask.iloc[j] = False  # 标记为删除
        
        # 返回保留的设计
        return group[keep_mask]

    # 应用优化后的函数到每个组
    results = []
    for algo_name, group in df.groupby('algo_name'):
        optimized_group = remove_invalid_designs_optimized(group)
        results.append(optimized_group)
    
    # 合并结果
    if results:
        df = pd.concat(results, ignore_index=True)
    else:
        df = pd.DataFrame(columns=df.columns)

    print(f"After deleting overlapping cases")
    print_info(df)
    return df

def add_is_pareto(df):
    if df.empty:
        print("DataFrame为空，无法添加Pareto信息")
        return df
        
    # 初始化is_pareto列
    df['is_pareto'] = False
    
    # 并行处理每个算法组
    def process_algo_group(group_data):
        group = group_data.copy()
        goal1 = group['ResourceMetric'].values
        goal2 = group['Best-caseLatency'].values
        is_pareto = [True] * len(goal1)
        
        # 优化的Pareto前沿计算
        for i in range(len(goal1)):
            for j in range(len(goal1)):
                if i != j:  # 不与自己比较
                    if (goal1[j] <= goal1[i] and goal2[j] <= goal2[i]) and (goal1[j] < goal1[i] or goal2[j] < goal2[i]):
                        is_pareto[i] = False
                        break
        
        group['is_pareto'] = is_pareto
        return group
    
    # 按算法名分组并处理
    processed_groups = []
    for algo_name, group in df.groupby('algo_name'):
        processed_group = process_algo_group(group)
        processed_groups.append(processed_group)
    
    # 合并处理后的组
    df = pd.concat(processed_groups, ignore_index=True)
    print(f"Successfully added 'is_pareto' column")
    return df

def embed_source_code(df):
    if df.empty:
        print("DataFrame为空，无法嵌入源代码")
        return df
        
    # 创建一个缓存来存储已处理的文件路径
    source_code_cache = {}
    
    def get_source_code(file_path):
        # 检查缓存
        if file_path in source_code_cache:
            return source_code_cache[file_path]
            
        base_dir = Path(file_path).parents[4]
        source_list = []
        
        for root, dirs, files in os.walk(base_dir):
            if '.autopilot' in dirs:
                dirs.remove('.autopilot')
            for fname in files:
                if fname.endswith((".c", ".cpp", ".h", ".hpp")):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r") as sf:
                            content = sf.read()
                        source_list.append({
                            "file_name": fname,
                            "file_content": content
                        })
                    except Exception as e:
                        print(f"Error reading {fpath}: {e}")
        
        # 保存到缓存
        source_code_cache[file_path] = source_list
        return source_list
    
    # 使用并行处理来加速源代码读取
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        source_codes = list(executor.map(get_source_code, df['File Path']))
    
    df['source_code'] = source_codes
    
    # 计算代码长度 - 只计算源代码文件的长度
    df['code_length'] = [sum([len(file['file_content']) for file in sc if file['file_name'].endswith((".c", ".cpp"))]) for sc in source_codes]

    print(f"Successfully embedded source code")
    return df

def add_is_kernel(df):
    if df.empty:
        print("DataFrame为空，无法添加kernel信息")
        return df
        
    # 使用向量化操作
    df['is_kernel'] = df['File Path'].astype(str).str.contains('kernels')
    print(f"Successfully added 'is_kernel' column")
    return df

def add_pragma_number(df):
    if df.empty:
        print("DataFrame为空，无法添加pragma数量")
        return df
        
    # 编译正则表达式模式以提高性能
    pragma_pattern = re.compile(r'#pragma.*')
    
    def get_pragma_number(source_code):
        pragma_number = 0
        for file in source_code:
            content = file['file_content']
            pragma_number += len(pragma_pattern.findall(content))
        return pragma_number
    
    # 使用并行处理计算pragma数量
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        pragma_numbers = list(executor.map(get_pragma_number, df['source_code']))
    
    df['pragma_number'] = pragma_numbers
    print(f"Successfully added 'pragma_number' column")
    return df

def analysis(df, save_path, dataset_name):
    if df.empty:
        print("DataFrame为空，无法生成分析")
        analysis_df = pd.DataFrame(columns=['algo_name', 'source_name', 'pragma_number', 'design_number', 'code_length'])
    else:
        # 使用groupby的agg函数一次性计算所有需要的统计信息
        analysis_df = df.groupby('algo_name').agg({
            'source_name': 'first',  # 取第一个source_name
            'pragma_number': 'first',  # 取第一个pragma_number
            'code_length': 'first'  # 取第一个code_length
        }).reset_index()
        
        # 添加设计数量列
        analysis_df['design_number'] = df.groupby('algo_name').size().values
        
        # 按algo_name排序
        analysis_df = analysis_df.sort_values('algo_name')
    
    # 保存分析结果
    analysis_csv_path = os.path.join(save_path, f'analysis_of_designs_{dataset_name}.csv')
    analysis_df.to_csv(analysis_csv_path, index=False)
    print(f"Successfully saved 'analysis_of_designs' to {analysis_csv_path}")

def add_top_function_name(df):
    if df.empty:
        print("DataFrame为空，无法添加顶层函数名")
        return df
        
    # 创建缓存存储已处理的路径
    top_function_cache = {}
    
    def get_top_function_name(file_path):
        # 检查缓存
        if file_path in top_function_cache:
            return top_function_cache[file_path]
            
        try:
            # 获取.xml文件的上一级目录
            xml_dir = os.path.dirname(file_path)
            # 获取上四级父目录
            top_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(xml_dir))))
            # 获取top_function_name.txt文件
            top_function_name_path = os.path.join(top_dir, 'top_function_name.txt')
            
            if os.path.exists(top_function_name_path):
                with open(top_function_name_path, 'r') as f:
                    top_function_name = f.read().strip()
                    # 保存到缓存
                    top_function_cache[file_path] = top_function_name
                    return top_function_name
            else:
                print(f"找不到顶层函数名文件: {top_function_name_path}")
                return "unknown"
        except Exception as e:
            print(f"获取顶层函数名时出错: {file_path}, 错误: {str(e)}")
            return "unknown"
    
    # 使用并行处理获取顶层函数名
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        top_functions = list(executor.map(get_top_function_name, df['File Path']))
    
    df['top_function_name'] = top_functions
    return df

def dataset_csv(search_path, save_path):
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    dataset_name = os.path.basename(search_path.rstrip('/'))
    print(f"dataset_name: {dataset_name}")
    raw_data_csv_path = os.path.join(save_path, f'raw_data_of_designs_{dataset_name}.csv')
    
    # 如果搜索路径不存在，则创建一个空的CSV
    if not os.path.exists(search_path):
        print(f"搜索路径不存在: {search_path}")
        with open(raw_data_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['File Path', 'Part', 'TargetClockPeriod', 'Best-caseLatency', 'Worst-caseLatency', 
                                   'BRAM_18K', 'LUT', 'DSP', 'FF', 'Avialable_BRAM_18K', 'Avialable_LUT', 'Avialable_DSP', 'Avialable_FF'])
        print(f"已创建空的原始数据CSV: {raw_data_csv_path}")
        return
    
    # 收集设计数据
    gather_csynth_data(search_path, raw_data_csv_path)

    print(f"read csv from {raw_data_csv_path}")
    df = pd.read_csv(raw_data_csv_path)
    
    # 检查DataFrame是否为空
    if df.empty:
        print("读取的CSV文件为空，将跳过后续处理")
        return
    
    # 处理设计数据
    df = base_feature(df)
    df = embed_source_code(df)
    df = add_pragma_number(df)

    # 生成分析数据
    analysis(df, save_path, dataset_name)

    # 数据清洗和标记
    df = delete_undef(df)
    if not df.empty:
        df = delete_overlap_and_overfitting(df)
        df = add_is_pareto(df)
        df = add_is_kernel(df)
        df = add_top_function_name(df)

        # 保存到JSON
        output_json_path = os.path.join(save_path, f'data_of_designs_{dataset_name}.json')
        df.to_json(output_json_path, orient='records', lines=True, force_ascii=False)
        print(f"The JSON file is saved in {output_json_path}")
    else:
        print("数据处理后为空，不生成最终JSON文件")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='处理HLS设计数据')
    parser.add_argument('--search_path', type=str, default='../data/designs/MachSuite',
                        help='设计数据的搜索路径')
    parser.add_argument('--save_path', type=str, default='./csv/',
                        help='保存结果的路径')
    
    args = parser.parse_args()
    
    dataset_csv(args.search_path, args.save_path)