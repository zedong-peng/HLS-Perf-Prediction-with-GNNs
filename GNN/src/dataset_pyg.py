import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


class GraphDataReader:
    """简化的图数据读取器（仅读取必要文件）"""
    
    @staticmethod
    def read_csv_graph_raw(raw_dir, add_inverse_edge=False):
        """从原始CSV文件读取图数据（仅必要字段）。"""
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
            else:
                graph['edge_index'] = edge[:, num_edge_accum:num_edge_accum+num_edge]
                
                if edge_feat is not None:
                    graph['edge_feat'] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
                else:
                    graph['edge_feat'] = None
            
            num_edge_accum += num_edge
            
            # 处理节点
            if node_feat is not None:
                graph['node_feat'] = node_feat[num_node_accum:num_node_accum+num_node]
            else:
                graph['node_feat'] = None
            
            graph['num_nodes'] = num_node
            num_node_accum += num_node
            
            graph_list.append(graph)
        
        return graph_list


class GraphConverter:
    """简化的图数据转换器（仅必要字段）"""
    
    @staticmethod
    def convert_to_pyg(graph_list):
        """将原始图数据转换为PyTorch Geometric格式。"""
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
            # 不再依赖 meta 文件，直接在此处定义默认元信息
            self.download_name = self.name
            self.num_tasks = 1
            self.eval_metric = 'rmse'
            self.__num_classes__ = -1
            self.binary = False
            self._default_split = 'scaffold'
            self._add_inverse_edge = False
            self._additional_node_files = []
            self._additional_edge_files = []
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            # 从传入的 meta_dict 读取（主要用于调试/兼容）
            self.download_name = meta_dict.get('download_name', self.name)
            self.num_tasks = int(meta_dict.get('num tasks', 1))
            self.eval_metric = meta_dict.get('eval metric', 'rmse')
            self.__num_classes__ = int(meta_dict.get('num classes', -1))
            self.binary = bool(meta_dict.get('binary', False))
            self._default_split = meta_dict.get('split', 'scaffold')
            self._add_inverse_edge = bool(meta_dict.get('add_inverse_edge', False))
            self._additional_node_files = []
            self._additional_edge_files = []
        
        # 若使用默认路径，以上属性已在上面设定；此处不再从 meta 读取
        
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
            # 所有数据集均包含节点与边特征
            return ['edge.csv.gz', 'node-feat.csv.gz', 'edge-feat.csv.gz']
    
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
            split_type = getattr(self, '_default_split', 'scaffold')
        
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
    
    # 额外文件解析已移除（不再使用）
    
    def process(self):
        """处理原始数据并转换为PyTorch Geometric格式"""
        print('开始处理数据集...')
        
        # 获取处理参数（统一默认值）
        add_inverse_edge = getattr(self, '_add_inverse_edge', False)
        # 读取原始图数据
        graph_list = GraphDataReader.read_csv_graph_raw(self.raw_dir, add_inverse_edge=add_inverse_edge)

        # 转换为PyTorch Geometric格式
        data_list = GraphConverter.convert_to_pyg(graph_list)
        
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
        
        # 简化为统一的回归任务处理（与集中配置一致）
        if self.binary:
            graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
        else:
            graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'),
                                      compression='gzip', header=None).values

        for i, g in enumerate(data_list):
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
    # 轻量级自检
    dataset = PygGraphPropPredDataset(name='dfg_dsp')
    split_idx = dataset.get_idx_split()
    print(f"Loaded dataset '{dataset.name}'. Train/Valid/Test: {len(split_idx['train'])}/{len(split_idx['valid'])}/{len(split_idx['test'])}")
