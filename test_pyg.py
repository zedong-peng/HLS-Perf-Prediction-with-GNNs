import torch
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(f"✅ Dataset loaded: {dataset}")
