import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import shutil
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


class GraphDataReader:
    """图数据读取器，负责从原始CSV文件读取图数据"""
    
    @staticmethod
    def read_csv_graph_raw(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[]):
        """
        从原始CSV文件读取图数据
        
        Args:
            raw_dir: 原始数据目录路径
            add_inverse_edge (bool): 是否添加反向边
            additional_node_files: 额外的节点文件列表
            additional_edge_files: 额外的边文件列表
            
        Returns:
            graph_list: 图列表，每个图是包含edge_index, edge_feat, node_feat, num_nodes的字典
        """
        print('Loading necessary files...')
        print('This might take a while.')
        
        # 加载必需文件
        try:
            edge = pd.read_csv(osp.join(raw_dir, 'edge.csv.gz'), 
                             compression='gzip', header=None).values.T.astype(np.int64)
            num_node_list = pd.read_csv(osp.join(raw_dir, 'num-node-list.csv.gz'), 
                                      compression='gzip', header=None).astype(np.int64)[0].tolist()
            num_edge_list = pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv.gz'), 
                                      compression='gzip', header=None).astype(np.int64)[0].tolist()
        except FileNotFoundError:
            raise RuntimeError('Missing necessary files (edge.csv.gz, num-node-list.csv.gz, num-edge-list.csv.gz)')
        
        # 加载节点特征
        try:
            node_feat = pd.read_csv(osp.join(raw_dir, 'node-feat.csv.gz'), 
                                  compression='gzip', header=None).values
            node_feat = node_feat.astype(np.int64 if 'int' in str(node_feat.dtype) else np.float32)
        except FileNotFoundError:
            node_feat = None
        
        # 加载边特征
        try:
            edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv.gz'), 
                                  compression='gzip', header=None).values
            edge_feat = edge_feat.astype(np.int64 if 'int' in str(edge_feat.dtype) else np.float32)
        except FileNotFoundError:
            edge_feat = None
        
        # 加载额外的节点信息
        additional_node_info = {}
        for additional_file in additional_node_files:
            assert additional_file.startswith('node_'), f"节点文件名必须以'node_'开头: {additional_file}"
            
            # 处理ogbn-proteins的特殊情况
            if additional_file == 'node_species' and osp.exists(osp.join(raw_dir, 'species.csv.gz')):
                os.rename(osp.join(raw_dir, 'species.csv.gz'), 
                         osp.join(raw_dir, 'node_species.csv.gz'))
            
            temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), 
                             compression='gzip', header=None).values
            additional_node_info[additional_file] = temp.astype(
                np.int64 if 'int' in str(temp.dtype) else np.float32)
        
        # 加载额外的边信息
        additional_edge_info = {}
        for additional_file in additional_edge_files:
            assert additional_file.startswith('edge_'), f"边文件名必须以'edge_'开头: {additional_file}"
            
            temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), 
                             compression='gzip', header=None).values
            additional_edge_info[additional_file] = temp.astype(
                np.int64 if 'int' in str(temp.dtype) else np.float32)
        
        # 构建图列表
        graph_list = []
        num_node_accum = 0
        num_edge_accum = 0
        
        print('Processing graphs...')
        for num_node, num_edge in tqdm(zip(num_node_list, num_edge_list), total=len(num_node_list)):
            graph = {}
            
            # 处理边
            if add_inverse_edge:
                # 复制边并添加反向边
                duplicated_edge = np.repeat(edge[:, num_edge_accum:num_edge_accum+num_edge], 2, axis=1)
                duplicated_edge[0, 1::2] = duplicated_edge[1, 0::2]
                duplicated_edge[1, 1::2] = duplicated_edge[0, 0::2]
                graph['edge_index'] = duplicated_edge
                
                if edge_feat is not None:
                    graph['edge_feat'] = np.repeat(edge_feat[num_edge_accum:num_edge_accum+num_edge], 2, axis=0)
                else:
                    graph['edge_feat'] = None
                
                for key, value in additional_edge_info.items():
                    graph[key] = np.repeat(value[num_edge_accum:num_edge_accum+num_edge], 2, axis=0)
            else:
                graph['edge_index'] = edge[:, num_edge_accum:num_edge_accum+num_edge]
                
                if edge_feat is not None:
                    graph['edge_feat'] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
                else:
                    graph['edge_feat'] = None
                
                for key, value in additional_edge_info.items():
                    graph[key] = value[num_edge_accum:num_edge_accum+num_edge]
            
            num_edge_accum += num_edge
            
            # 处理节点
            if node_feat is not None:
                graph['node_feat'] = node_feat[num_node_accum:num_node_accum+num_node]
            else:
                graph['node_feat'] = None
            
            for key, value in additional_node_info.items():
                graph[key] = value[num_node_accum:num_node_accum+num_node]
            
            graph['num_nodes'] = num_node
            num_node_accum += num_node
            
            graph_list.append(graph)
        
        return graph_list


class GraphConverter:
    """图数据转换器，负责将原始图数据转换为PyTorch Geometric格式"""
    
    @staticmethod
    def convert_to_pyg(graph_list, additional_node_files=[], additional_edge_files=[]):
        """
        将原始图数据转换为PyTorch Geometric格式
        
        Args:
            graph_list: 原始图数据列表
            additional_node_files: 额外的节点文件列表
            additional_edge_files: 额外的边文件列表
            
        Returns:
            pyg_graph_list: PyTorch Geometric图对象列表
        """
        pyg_graph_list = []
        
        print('Converting graphs into PyG objects...')
        
        for graph in tqdm(graph_list):
            g = Data()
            g.__num_nodes__ = graph['num_nodes']
            g.edge_index = torch.from_numpy(graph['edge_index'])
            
            # 处理边特征
            if graph['edge_feat'] is not None:
                g.edge_attr = torch.from_numpy(graph['edge_feat'])
            
            # 处理节点特征
            if graph['node_feat'] is not None:
                g.x = torch.from_numpy(graph['node_feat'])
            
            # 处理额外的节点文件
            for key in additional_node_files:
                if key in graph and graph[key] is not None:
                    g[key] = torch.from_numpy(graph[key])
            
            # 处理额外的边文件
            for key in additional_edge_files:
                if key in graph and graph[key] is not None:
                    g[key] = torch.from_numpy(graph[key])
            
            pyg_graph_list.append(g)
        
        return pyg_graph_list


class PygGraphPropPredDataset(InMemoryDataset):
    """PyTorch Geometric图属性预测数据集"""
    
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        """
        初始化数据集
        
        Args:
            name (str): 数据集名称
            root (str): 存储数据集的根目录
            transform: 图变换函数
            pre_transform: 预处理变换函数
            meta_dict: 元信息字典，用于调试或外部贡献者
        """
        self.name = name
        
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))
            
            # 检查是否存在之前下载的文件夹
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'
            
            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            # 读取master.csv获取元信息
            master_path = osp.join(osp.dirname(__file__), 'master.csv')
            master = pd.read_csv(master_path, index_col=0)
            
            if self.name not in master:
                error_mssg = f'Invalid dataset name {self.name}.\n'
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            
            self.meta_info = master[self.name]
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict
        
        # 设置数据集属性
        self.download_name = self.meta_info['download_name']
        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'
        
        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def num_classes(self):
        """返回类别数量"""
        return self.__num_classes__
    
    @property
    def raw_file_names(self):
        """返回原始文件名列表"""
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]
    
    @property
    def processed_file_names(self):
        """返回处理后的文件名"""
        return 'geometric_data_processed.pt'
    
    def get_idx_split(self, split_type=None):
        """
        获取数据集划分索引
        
        Args:
            split_type: 划分类型，默认使用meta_info中的split
            
        Returns:
            split_dict: 包含train, valid, test索引的字典
        """
        if split_type is None:
            split_type = self.meta_info['split']
        
        path = osp.join(self.root, 'split', split_type)
        
        # 如果split_dict.pt存在，直接加载
        split_dict_path = osp.join(path, 'split_dict.pt')
        if osp.isfile(split_dict_path):
            return torch.load(split_dict_path, weights_only=False)
        
        split_dict = {}
        
        # 加载各个划分的索引
        for split_name in ['train', 'valid', 'test']:
            split_file_path = osp.join(path, f'{split_name}.csv.gz')
            
            if osp.exists(split_file_path):
                try:
                    idx = pd.read_csv(split_file_path, compression='gzip', header=None).values.T[0]
                    split_dict[split_name] = torch.tensor(idx, dtype=torch.long)
                except Exception as e:
                    print(f"警告：读取{split_name}集索引时出错: {e}")
                    split_dict[split_name] = torch.tensor([], dtype=torch.long)
            else:
                split_dict[split_name] = torch.tensor([], dtype=torch.long)
        
        return split_dict
    
    def _parse_additional_files(self, additional_files_str):
        """解析额外文件字符串"""
        if additional_files_str == 'None' or pd.isna(additional_files_str):
            return []
        return str(additional_files_str).split(',')
    
    def process(self):
        """处理原始数据并转换为PyTorch Geometric格式"""
        print('开始处理数据集...')
        
        # 获取处理参数
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'
        additional_node_files = self._parse_additional_files(self.meta_info.get('additional node files'))
        additional_edge_files = self._parse_additional_files(self.meta_info.get('additional edge files'))
        
        # 读取原始图数据
        graph_list = GraphDataReader.read_csv_graph_raw(
            self.raw_dir, 
            add_inverse_edge=add_inverse_edge,
            additional_node_files=additional_node_files,
            additional_edge_files=additional_edge_files
        )
        
        # 转换为PyTorch Geometric格式
        data_list = GraphConverter.convert_to_pyg(
            graph_list, 
            additional_node_files=additional_node_files,
            additional_edge_files=additional_edge_files
        )
        
        # 处理标签
        self._process_labels(data_list)
        
        # 应用预处理变换
        if self.pre_transform is not None:
            print('应用预处理变换...')
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]
        
        # 整理数据并保存
        print('整理数据并保存...')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f'数据集处理完成，已保存到 {self.processed_paths[0]}')
    
    def _process_labels(self, data_list):
        """处理图标签"""
        print('处理图标签...')
        
        if self.task_type == 'subtoken prediction':
            # 处理子词预测任务
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), 
                                              compression='gzip', header=None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]
            
            for i, g in enumerate(data_list):
                g.y = graph_label[i]
        else:
            # 处理其他任务类型
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), 
                                        compression='gzip', header=None).values
            
            has_nan = np.isnan(graph_label).any()
            
            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)


def create_dataset(name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
    """
    创建数据集的便利函数
    
    Args:
        name (str): 数据集名称
        root (str): 根目录
        transform: 变换函数
        pre_transform: 预处理变换函数
        meta_dict: 元信息字典
        
    Returns:
        PygGraphPropPredDataset: 数据集对象
    """
    return PygGraphPropPredDataset(name, root, transform, pre_transform, meta_dict)


if __name__ == '__main__':
    # 示例用法
    print("测试数据集加载...")
    
    # 创建数据集
    dataset = PygGraphPropPredDataset(name='dfg_dsp')
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {dataset.num_classes}")
    print(f"任务类型: {dataset.task_type}")
    print(f"评估指标: {dataset.eval_metric}")
    
    # 获取数据划分
    split_idx = dataset.get_idx_split()
    print(f"训练集大小: {len(split_idx['train'])}")
    print(f"验证集大小: {len(split_idx['valid'])}")
    print(f"测试集大小: {len(split_idx['test'])}")
    
    # 查看第一个图
    if len(dataset) > 0:
        first_graph = dataset[0]
        print(f"第一个图的节点数: {first_graph.num_nodes}")
        print(f"第一个图的边数: {first_graph.edge_index.size(1)}")
        if hasattr(first_graph, 'x') and first_graph.x is not None:
            print(f"节点特征维度: {first_graph.x.shape}")
        if hasattr(first_graph, 'edge_attr') and first_graph.edge_attr is not None:
            print(f"边特征维度: {first_graph.edge_attr.shape}")
    
    # 测试数据加载器
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print("\n测试批处理加载...")
    for batch in loader:
        print(f"批次大小: {batch.num_graphs}")
        print(f"批次节点总数: {batch.num_nodes}")
        print(f"批次边总数: {batch.edge_index.size(1)}")
        break

