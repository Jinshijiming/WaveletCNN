"""
"STF-GCN: A Multi-Domain Graph Convolution Network Method for Automatic Modulation Recognition via Adaptive Correlation"
official implementation
"""
from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch_geometric.nn.dense import DenseSAGEConv, dense_diff_pool, DenseGATConv
from thop import profile, clever_format
import time

from .. import BaseModel


class Fusion(nn.Module):

    def __init__(self, input_chanel, drop=0.):

        super().__init__()

        self.Conv = nn.Sequential(nn.Conv1d(input_chanel, input_chanel, 1),
                                  nn.BatchNorm1d(input_chanel),
                                  nn.GELU(),
                                  nn.Conv1d(input_chanel, input_chanel, 1),
                                  )
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, stft):
        fusion = self.Conv(torch.cat((x, stft), dim=1))
        return self.drop(fusion)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=True, lin=True):
        super(GNN, self).__init__()
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv4 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn4 = torch.nn.BatchNorm1d(out_channels)
        if lin is True:
            self.lin = torch.nn.Linear(3 * hidden_channels + out_channels, out_channels)
        else:
            self.lin = None

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
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))
        x4 = self.bn(4, F.relu(self.conv4(x3, adj, mask)))
        x = torch.cat([x1, x2, x3, x4], dim=-1)
        if self.lin is not None:
            x = F.relu(self.lin(x))
        return x


class AGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=True, lin=True):
        super(AGNN, self).__init__()
        self.gat1 = DenseGATConv(in_channels, hidden_channels, 1, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.gat2 = DenseGATConv(hidden_channels, out_channels, 1, normalize)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

    def bn(self, i, x):
        batch_size_, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size_, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, F.relu(self.gat1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.gat2(x1, adj, mask)))
        x = torch.cat([x1, x2], dim=-1)
        return x


class Net(BaseModel):
    def __init__(self, in_len=128, num_filter=11, num_classes=11):
        super(Net, self).__init__()

        self.num_filter = num_filter

        self.hid = 32

        self.bniq = nn.BatchNorm1d(2)
        self.iq_conv1d = nn.Conv1d(in_channels=2, out_channels=self.hid // 2, kernel_size=3, stride=1, padding=1,
                                   groups=2)
        self.bnstft = nn.BatchNorm2d(1)
        self.stft_conv2d = nn.Conv2d(in_channels=1, out_channels=self.hid // 2, kernel_size=(40, 1))
        self.fusion = Fusion(self.hid)
        self.LSTM = nn.LSTM(input_size=self.hid, hidden_size=self.hid, bidirectional=True, batch_first=True,
                            num_layers=2, dropout=0.2)
        self.pool = nn.MaxPool1d(4, 4)

        self.globalmaxpool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )

        num_nodes = ceil(0.25 * in_len)
        self.gnn1_pool = GNN(self.hid // 2, self.hid * 2, num_nodes)
        self.gnn1_embed = GNN(self.hid // 2, self.hid * 2, self.hid * 2, lin=False)
        self.gat1_embed = AGNN(self.hid * 2 * 4, self.hid * 2 * 2, self.hid * 2 * 2)
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(self.hid * 2 * 4, self.hid * 2, num_nodes)
        self.gnn2_embed = GNN(self.hid * 2 * 4, self.hid * 2, self.hid * 2, lin=False)
        self.gat2_embed = AGNN(self.hid * 2 * 4, self.hid * 2 * 2, self.hid * 2 * 2)
        self.gnn3_embed = GNN(self.hid * 2 * 4, self.hid * 2, self.hid * 2, lin=False)

        self.globalavgpool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.linear = torch.nn.Linear(self.hid * 2 * 4, num_classes)

    def adj_generate(self, x):
        adj = torch.zeros(x.shape[0], x.shape[1], x.shape[1], dtype=torch.float, device=x.device)
        for i in range(self.num_filter):
            other_x_i = x[:, 0:-1 - i, :] * x[:, i + 1:, :]
            other_x_i = self.globalmaxpool(other_x_i)
            other_x_i = F.relu(other_x_i, inplace=True)
            other_x_i = torch.diag_embed(other_x_i, offset=(i + 1))
            adj = torch.add(adj, other_x_i)
        adj_2 = adj.permute(0, 2, 1)
        adj = adj + adj_2

        return adj

    def forward(self, x_in, mask=None):
        x_in_iq = x_in[:, :2, :]
        x_iq = self.bniq(x_in_iq)
        x_iq = self.iq_conv1d(x_iq)
        x_in_stft = x_in[:, 2:, :]
        x_stft = x_in_stft.unsqueeze(1)
        x_stft = self.bnstft(x_stft)
        x_stft = self.stft_conv2d(x_stft).squeeze(2)
        x = self.fusion(x_iq, x_stft)
        x, _ = self.LSTM(x.permute(0, 2, 1))
        x = self.pool(x)

        adj = self.adj_generate(x)

        s_1 = self.gnn1_pool(x, adj, mask)
        x_1 = self.gnn1_embed(x, adj, mask)
        x_1, adj_1, _, _ = dense_diff_pool(x_1, adj, s_1, mask)
        x_1 = self.gat1_embed(x_1, adj_1)

        s_2 = self.gnn2_pool(x_1, adj_1)
        x_2 = self.gnn2_embed(x_1, adj_1)
        x_2, adj_2, _, _ = dense_diff_pool(x_2, adj_1, s_2)
        x_2 = self.gat2_embed(x_2, adj_2)

        x_3 = self.gnn3_embed(x_2, adj_2)
        x_cls = self.linear(self.globalavgpool(x_3.permute(0, 2, 1)))

        return x_cls



if __name__ == '__main__':
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    model = Net('2016.10a', device).to(device)
    x = torch.randn(2, 42, 128).to(device)
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
