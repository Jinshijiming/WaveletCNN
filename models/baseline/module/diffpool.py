import torch
import torch.nn.functional as F
from math import ceil
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


class GNN(torch.nn.Module):
    def __init__(self, in_channel, hidden_channels, out_channels, g_layers=6, lin=False, Gcn=DenseSAGEConv):
        super().__init__()

        self.gconvs = torch.nn.ModuleList()
        self.gbns = torch.nn.ModuleList()

        self.gconvs.append(Gcn(in_channel, hidden_channels))
        self.gbns.append(torch.nn.BatchNorm1d(hidden_channels))
        for i in range(g_layers - 2):
            self.gconvs.append(Gcn(hidden_channels, hidden_channels))
            self.gbns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.gconvs.append(Gcn(hidden_channels, out_channels))
        self.gbns.append(torch.nn.BatchNorm1d(out_channels))

        if lin is True:
            self.lin = torch.nn.Linear((g_layers - 1) * hidden_channels + out_channels, out_channels)
        else:  # 修改diffpool的全连接层
            self.lin = None

    def forward(self, x, adj):
        temp = []
        for gconv, gbn in zip(self.gconvs, self.gbns):
            x = F.relu(gconv(x, adj))
            x = gbn(x.permute(0, 2, 1)).permute(0, 2, 1)
            temp.append(x)
        x = torch.cat(temp, dim=-1)
        if self.lin is not None:
            x = F.relu(self.lin(x))
        return x


class diff_pool_layer(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, pool_rate, g_layers, lin=False):
        torch.nn.Module.__init__(self)  ##参数不同所以需要分开初始化

        self.num_nodes = ceil(pool_rate * num_nodes)
        self.gnn1_pool = GNN(in_channels, hidden_channels, self.num_nodes, g_layers=g_layers, lin=True)
        self.gnn1_embed = GNN(in_channels, hidden_channels, hidden_channels, g_layers=g_layers, lin=lin)

    def forward(self, x, adj):
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)

        return x, adj, l1, e1


class DiffPools(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, d_layer=2, pool_rate=0.75, g_layers=6, lin=False):
        torch.nn.Module.__init__(self)  ##参数不同所以需要分开初始化
        self.diffPools = torch.nn.ModuleList()
        self.diffPools.append(diff_pool_layer(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            pool_rate=pool_rate,
            g_layers=g_layers
        ))
        for i in range(d_layer - 2):
            num_nodes = ceil(pool_rate * num_nodes)
            self.diffPools.append(diff_pool_layer(
                num_nodes=num_nodes,
                in_channels=g_layers * hidden_channels,  # 中间diff层输出维度为g_layers * hidden_channels
                hidden_channels=hidden_channels,
                pool_rate=pool_rate,
                g_layers=g_layers
            ))
            # print(1)

        num_nodes = ceil(pool_rate * num_nodes)
        self.diffPools.append(diff_pool_layer(
            num_nodes=num_nodes,
            in_channels=g_layers * hidden_channels,
            hidden_channels=hidden_channels,
            pool_rate=pool_rate,
            g_layers=g_layers, lin=lin  # lin=True的话最后一个diff层输出维度为hidden_channels
        ))

    def forward(self, x, adj):
        for diffpool in self.diffPools:
            # print(x.shape)
            x, adj, l, e = diffpool(x, adj)
        return x, adj, l, e


if __name__ == '__main__':
    A = GNN(64, 64, 64)
    print(A)
