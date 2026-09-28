"""
Knowledge Embedded Convolutional Transformer Hybrid Network for Automatic Modulation Classification
https://ieeexplore.ieee.org/abstract/document/11143389

reproduced by asme
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import BaseModel


class ArFAMCNet(BaseModel):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ArFAMCNet, self).__init__()

        # 特征提取输出11维特征
        self.feature_dim = 11
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

        # 使用He初始化
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def extract_features(self, x):
        """
        从原始I/Q数据中提取手工特征 (PyTorch向量化实现)
        输入: x shape (batch_size, 2, signal_length) - I/Q数据
        输出: features shape (batch_size, 11) - 提取的特征
        """
        batch_size = x.size(0)

        # 提取I/Q分量
        I_data = x[:, 0, :]  # (batch_size, signal_length)
        Q_data = x[:, 1, :]  # (batch_size, signal_length)

        # 1. AP特征：幅度比特征 (向量化)
        amplitude = torch.sqrt(I_data ** 2 + Q_data ** 2)  # (batch_size, signal_length)
        amplitude_min = torch.min(amplitude, dim=1, keepdim=True)[0]  # (batch_size, 1)
        amplitude_max = torch.max(amplitude, dim=1, keepdim=True)[0]  # (batch_size, 1)

        # 防止除零
        amplitude_min = torch.where(amplitude_min > 0, amplitude_min, torch.ones_like(amplitude_min))
        Param_R = (amplitude_max / amplitude_min).squeeze()  # (batch_size,)

        # 2. 复数数据 (向量化)
        comp_data = torch.complex(I_data, Q_data)  # (batch_size, signal_length)
        comp_data_conj = torch.conj(comp_data)

        # 3. 计算各阶矩 (向量化实现)
        # 一阶矩
        mom_10 = torch.mean(comp_data, dim=1)  # (batch_size,)

        # 二阶矩
        mom_20 = torch.mean(comp_data ** 2, dim=1)
        mom_21 = torch.mean(comp_data * comp_data_conj, dim=1)  # |z|^2的均值

        # 三阶矩
        mom_30 = torch.mean(comp_data ** 3, dim=1)

        # 四阶矩
        mom_40 = torch.mean(comp_data ** 4, dim=1)
        mom_41 = torch.mean((comp_data ** 3) * comp_data_conj, dim=1)
        mom_42 = torch.mean((comp_data ** 2) * (comp_data_conj ** 2), dim=1)

        # 五阶矩
        mom_50 = torch.mean(comp_data ** 5, dim=1)

        # 六阶矩
        mom_60 = torch.mean(comp_data ** 6, dim=1)
        mom_63 = torch.mean((comp_data ** 3) * (comp_data_conj ** 3), dim=1)

        # 七阶矩
        mom_70 = torch.mean(comp_data ** 7, dim=1)

        # 八阶矩
        mom_80 = torch.mean(comp_data ** 8, dim=1)

        # 4. 计算高阶累积量 (向量化)
        C_20 = mom_20
        C_21 = mom_21
        C_40 = mom_40 - 3 * mom_20 ** 2
        C_41 = mom_41 - 3 * mom_20 * mom_21
        C_42 = mom_42 - mom_20 ** 2 - 2 * mom_21 ** 2
        C_60 = mom_60 - 15 * mom_40 * mom_20 + 30 * mom_20 ** 3
        C_63 = mom_63 - 9 * C_42 * C_21 - 6 * C_21 ** 3
        C_80 = mom_80 - 28 * mom_60 * mom_20 - 35 * mom_40 ** 2 + 420 * mom_40 * mom_20 ** 2 - 630 * mom_20 ** 4

        # 5. 构建特征参数 (向量化)
        # 添加小的epsilon防止除零
        # eps = torch.tensor(1e-8, device=x.device, dtype=C_21.dtype)

        M_1 = C_20 / (C_21)
        M_3 = C_40 / (C_42)
        M_2 = torch.abs(C_42 / (C_21 ** 2))
        M_4 = torch.abs(C_40 / (C_21 ** 2))
        M_5 = torch.abs(C_63 / (C_21 ** 3))
        M_6 = torch.abs(C_80 / (C_21 ** 4))

        # 6. 组装特征向量 (11维) - 向量化
        features = torch.stack([
            Param_R.real,  # 0: 幅度比
            M_1.real,  # 1: M_1实部
            M_1.imag,  # 2: M_1虚部
            M_2.real,  # 3: M_2 (已经是实数)
            M_3.real,  # 4: M_3实部
            M_3.imag,  # 5: M_3虚部
            M_4.real,  # 6: M_4 (已经是实数)
            M_5.real,  # 7: M_5 (已经是实数)
            M_6.real,  # 8: M_6 (已经是实数)
            C_60.real,  # 9: C_60实部
            C_60.imag  # 10: C_60虚部
        ], dim=1)  # (batch_size, 11)

        # # 处理NaN和Inf值
        # features = torch.where(torch.isnan(features),
        #                        torch.zeros_like(features),
        #                        features)
        # features = torch.where(torch.isinf(features),
        #                        torch.zeros_like(features),
        #                        features)

        return features

    def forward(self, x):
        """

        :param x: (batch_size, 2, signal_length)
        :return:
        """
        x = x.permute(0, 2, 1)
        # 如果输入是原始I/Q数据，先提取特征
        if x.size(-1) > self.feature_dim:  # 原始信号长度 > 特征维度
            x = self.extract_features(x)

        # 前向传播
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
