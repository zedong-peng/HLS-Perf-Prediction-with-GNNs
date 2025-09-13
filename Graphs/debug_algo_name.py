import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv('csv/raw_data_of_designs_CHStone.csv')

print("原始数据:")
print(f"总行数: {len(df)}")
print(f"Best-caseLatency列的值: {df['Best-caseLatency'].tolist()}")
print(f"undef值的数量: {sum(df['Best-caseLatency'] == 'undef')}")
print(f"有效值的数量: {sum(df['Best-caseLatency'] != 'undef')}")

# 提取路径信息
df['File Path'] = df['File Path'].astype(str)
path_parts = df['File Path'].str.split('/')

print("\n路径分析:")
for i, path in enumerate(df['File Path']):
    parts = path.split('/')
    print(f"行 {i}: {path}")
    print(f"  总长度: {len(parts)}")
    print(f"  -8: {parts[-8] if len(parts) >= 8 else 'N/A'}")
    print(f"  -7: {parts[-7] if len(parts) >= 7 else 'N/A'}")
    print(f"  -6: {parts[-6] if len(parts) >= 6 else 'N/A'}")
    print()

# 应用提取逻辑
max_parts = path_parts.str.len().max()
print(f"最大路径长度: {max_parts}")

# 根据路径是否包含kernels来使用不同的解析逻辑
is_kernels = df['File Path'].str.contains('kernels')
print(f"是否包含kernels: {is_kernels.tolist()}")

# 初始化列
df['design_id'] = 'unknown'
df['algo_name'] = 'unknown'
df['source_name'] = 'unknown'

# 对于kernels路径: .../kernels/CHStone/aes/project/solution1/syn/report/csynth.xml
kernels_mask = is_kernels
if kernels_mask.any():
    if max_parts >= 5:
        df.loc[kernels_mask, 'design_id'] = path_parts[kernels_mask].str[-5]  # project
    if max_parts >= 6:
        df.loc[kernels_mask, 'algo_name'] = path_parts[kernels_mask].str[-6]  # aes
    if max_parts >= 7:
        df.loc[kernels_mask, 'source_name'] = path_parts[kernels_mask].str[-7]  # CHStone

# 对于designs路径: .../designs/CHStone/aes/design_01/project/solution1/syn/report/csynth.xml
designs_mask = ~is_kernels
if designs_mask.any():
    if max_parts >= 6:
        df.loc[designs_mask, 'design_id'] = path_parts[designs_mask].str[-6]  # design_01
    if max_parts >= 7:
        df.loc[designs_mask, 'algo_name'] = path_parts[designs_mask].str[-7]  # aes
    if max_parts >= 8:
        df.loc[designs_mask, 'source_name'] = path_parts[designs_mask].str[-8]  # CHStone

print(f"\n提取的algo_name:")
print(df['algo_name'].tolist())
print(f"唯一的algo_name: {df['algo_name'].unique()}")
print(f"算法数量: {df['algo_name'].nunique()}")

print(f"\n提取的source_name:")
print(df['source_name'].tolist())
print(f"唯一的source_name: {df['source_name'].unique()}")

print(f"\n提取的design_id:")
print(df['design_id'].tolist())
print(f"唯一的design_id: {df['design_id'].unique()}")

# 模拟delete_undef逻辑
print("\n模拟delete_undef逻辑:")
df_copy = df.copy()
df_copy['Best-caseLatency'] = df_copy['Best-caseLatency'].replace('undef', pd.NA)

valid_groups = []
for algo_name, group in df_copy.groupby('algo_name'):
    print(f"算法: {algo_name}")
    print(f"  组大小: {len(group)}")
    print(f"  Best-caseLatency: {group['Best-caseLatency'].tolist()}")
    print(f"  有NaN值: {group['Best-caseLatency'].isna().any()}")
    print(f"  会被保留: {not group['Best-caseLatency'].isna().any()}")
    print()
    
    if not group['Best-caseLatency'].isna().any():
        valid_groups.append(group)

print(f"有效组数量: {len(valid_groups)}") 