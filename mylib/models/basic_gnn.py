
import numpy as np
import torch as tc
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv


class Model(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = tc.device(device=device)
        self.conv1 = GCNConv(4, 12)
        self.conv2 = GCNConv(12, 24)
        self.conv3 = GCNConv(24, 48)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(48, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.lk = nn.LeakyReLU(0.1)
        pass

    def forward(self, node_features: tc.Tensor, edge_index: tc.Tensor, edge_weight:tc.Tensor) -> tc.Tensor:
        """
        Parameters:
        - x
        """
        node_features = self.conv1(node_features, edge_index, edge_weight)
        node_features = self.lk(node_features)
        node_features = self.conv2(node_features, edge_index, edge_weight)
        node_features = self.lk(node_features)
        node_features = self.conv3(node_features, edge_index, edge_weight)
        node_features = self.lk(node_features)
        x = self.flatten(node_features)
        x = x.sum(axis=0).reshape(1, -1)
        x = self.fc1(x)
        x = self.lk(x)
        x = self.fc2(x)
        x = self.lk(x)
        x = self.fc3(x)
        return x
    pass

