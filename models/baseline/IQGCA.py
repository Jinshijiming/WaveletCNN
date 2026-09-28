"""
"IQGCA: A Lightweight Graph Structure of I/Q Signal for Automatic Modulation Recognition"
https://ieeexplore.ieee.org/abstract/document/11091518

official implementation
"""
from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseSAGEConv, DenseGATConv, DMoNPooling
import math
from thop import profile, clever_format
from ..BaseModel import BaseModel

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=True, lin=True):
        super(GCN, self).__init__()
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

    def bn(self, i, x):
        batch_size_, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size_, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x = torch.cat([x1, x2], dim=-1)
        return x


class AGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(AGNN, self).__init__()
        self.gat1 = DenseGATConv(in_channels, out_channels, 1, normalize)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)

    def bn(self, i, x):
        batch_size_, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size_, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, F.relu(self.gat1(x, adj, mask)))

        return x


class Net(BaseModel):
    def __init__(self, in_len, num_classes):
        super(Net, self).__init__()

        max_nodes = in_len

        self.hid = 32

        self.bniq1 = nn.BatchNorm2d(1, eps=1e-5)
        self.iq_conv1d1 = nn.Conv1d(in_channels=2, out_channels=self.hid // 2, kernel_size=3, stride=1, padding=1,
                                    groups=2)
        self.iq_conv1d2 = nn.Conv1d(in_channels=self.hid // 2, out_channels=self.hid, kernel_size=3, stride=1,
                                    padding=1, groups=self.hid // 2)
        self.LSTM = nn.LSTM(input_size=self.hid, hidden_size=self.hid, bidirectional=False, batch_first=True,
                            num_layers=2, dropout=0.2)
        self.pool = nn.MaxPool1d(2, 2)

        num_nodes = ceil(0.25 * max_nodes)
        self.gcn1 = GCN(self.hid // 2, self.hid, self.hid)
        self.gpool1 = DMoNPooling([self.hid + self.hid, self.hid + self.hid], num_nodes)
        self.gat1 = AGNN(self.hid + self.hid, self.hid + self.hid)

        num_nodes = ceil(0.25 * num_nodes)
        self.gcn2 = GCN(self.hid + self.hid, self.hid, self.hid)
        self.gpool2 = DMoNPooling([self.hid + self.hid, self.hid + self.hid], num_nodes)
        self.gat2 = AGNN(self.hid + self.hid, self.hid + self.hid)

        self.globalavgpool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.linear = torch.nn.Linear(self.hid + self.hid, num_classes)

    def adj_generate(self, x_in):
        num_filter = int(math.sqrt(x_in.shape[2]))
        adj = torch.zeros((x_in.shape[0], x_in.shape[2], x_in.shape[2]), dtype=torch.float, device=self.device)
        for i in range(num_filter):
            other_x_i = x_in[:, 0, 0:-1 - i] * x_in[:, 0, i + 1:]
            other_x_i = torch.where(other_x_i > 0, torch.tensor(1.0, device=self.device),
                                    torch.tensor(0.0, device=self.device))
            other_x_i = other_x_i * x_in[:, 1, 0:-1 - i] * x_in[:, 1, i + 1:]
            other_x_i = torch.where(other_x_i > 0, torch.tensor(1.0, device=self.device),
                                    torch.tensor(0.0, device=self.device))
            other_x_i = torch.diag_embed(other_x_i, offset=(i + 1))
            adj = torch.add(adj, other_x_i)
        adj2 = adj.permute(0, 2, 1)
        adj = adj + adj2

        return adj

    def forward(self, x_in, mask=None):

        x_in= x_in.permute(0, 2, 1)  # B L C->B C L
        # 预处理I-Time和Q-Time信息，得到信号图结构节点特征
        x_in_iq = x_in.unsqueeze(1)
        x_iq = self.bniq1(x_in_iq)
        x_iq = x_iq.squeeze()
        x_iq = self.iq_conv1d1(x_iq)
        x_iq = self.iq_conv1d2(x_iq)
        x, _ = self.LSTM(x_iq.permute(0, 2, 1))
        x = self.pool(x)
        # 预处理I-Q信息，得到信号图结构边特征
        adj = self.adj_generate(x_in)
        # 提取信号图结构特征
        x_1 = self.gcn1(x, adj, mask)
        _, x_1, adj_1, _, _, _ = self.gpool1(x_1, adj, mask)
        x_1 = self.gat1(x_1, adj_1)
        x_2 = self.gcn2(x_1, adj_1)
        _, x_2, adj_2, _, _, _ = self.gpool2(x_2, adj_1)
        x_2 = self.gat2(x_2, adj_2)
        # 分类器
        x_cls = self.linear(self.globalavgpool(x_2.permute(0, 2, 1)))

        return x_cls


if __name__ == '__main__':
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = Net('2016.10a', device).to(device)
    x = torch.randn(2, 2, 128).to(device)
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
