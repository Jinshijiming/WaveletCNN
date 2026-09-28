import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        return A


# 注意力分数算子
class AttentionOperator(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None):
        super(AttentionOperator, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.n_heads = n_heads

    def forward(self, queries, keys):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)

        attn = self.inner_attention(
            queries,
            keys
        )

        return attn


class AttentionGraph(nn.Module):
    def __init__(self, d_input, d_key, top_k):
        super(AttentionGraph, self).__init__()
        self.queryProjection = nn.Linear(d_input, d_key)
        self.keyProjection = nn.Linear(d_input, d_key)
        self.valueProjection = nn.Linear(d_input, d_key)
        self.dropout = nn.Dropout(0.1)
        self.top_k = top_k

    def forward(self, x):
        queries = F.relu(self.queryProjection(x))
        keys = F.relu(self.keyProjection(x))
        # values=self.valueProjection(x)

        B, L, E = queries.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("ble,bse->bls", queries, keys)
        scores = torch.softmax(scale * scores, dim=-1)
        # scores = torch.relu(scale * scores)
        scores = self.keep_top_k(scores, self.top_k)
        scores = self.dropout(scores)
        return scores

    def keep_top_k(self, adjacency_matrix, k: int):
        # 找出每行最大的 k 个元素的值和索引
        top_values, top_indices = adjacency_matrix.topk(k, dim=-1)

        # 创建一个全零矩阵，大小与邻接矩阵相同
        masked_adjacency_matrix = torch.zeros_like(adjacency_matrix)

        # 使用索引和值将矩阵填充
        masked_adjacency_matrix.scatter_(-1, top_indices, top_values)

        return masked_adjacency_matrix


# AVG算子
class AvgOperator(nn.Module):
    def __init__(self, num_filter, num_channel=1):
        super(AvgOperator, self).__init__()
        self.num_filter = num_filter
        self.num_channel = num_channel
        self.add_convd = nn.ModuleList()

        for i in range(num_filter):
            if num_channel == 1:
                self.add_convd.append(
                    nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(i + 2) * num_channel)
                )  # 卷积核长度最大等于特征长度
            else:
                self.add_convd.append(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(i + 2, num_channel))
                )  # 卷积核长度最大等于特征长度

    def forward(self, x):
        """

        :param x: [B,L,F]
        :return:
        """
        adj = []
        if self.num_channel == 1:
            x = x.permute(0, 2, 1)  # [B,F=1,L]
        else:
            x = x.unsqueeze(dim=1)  # [B,C=1,L,F]

        for index, conv in enumerate(self.add_convd):
            other_x = conv(x)  # 【B,C=1,L,1】，一维卷积处理后的序列
            # other_x = torch.squeeze(other_x)  # 【B,L】
            other_x = F.relu(torch.squeeze(other_x), inplace=True)  # 【B,L】

            # other_x = torch.diag_embed(other_x, offset=(j + 1))  # 【B,L,L】
            # print(other_x.shape) 对角线需要小于等于 信号长度-2
            # print(other_x.shape)
            adj.append(torch.diag_embed(other_x, offset=(index + 1)))

        adj = torch.stack(adj)
        adj = torch.sum(adj, dim=0)  # 各个对角线矩阵依次相加【B*L，W,W】
        adj_array = adj + adj.permute(0, 2, 1)  # 【B*L，W,W】
        del adj
        return adj_array


def constellationOperator(x):
    # [B,L,C]

    if x.ndim == 3:
        x = torch.view_as_complex(x)

    distance = x.unsqueeze(1) - x.unsqueeze(2)
    adj = torch.abs(distance)
    adj = F.softmax(adj, dim=-1)
    return adj
