import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
import math
from .. import BaseModel
from einops.layers.torch import Rearrange
# ==========================================
# 1. 自适应小波卷积组件 (核心频率提取)
# ==========================================


class WaveletConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, length=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels  # 最终输出的通道数
        self.kernel_size = kernel_size
        self.length = length
        self.groups = groups
        self.stride = stride

        # 验证分组有效性
        assert out_channels % groups == 0, "out_channels 必须能被 groups 整除"
        assert in_channels % groups == 0, "in_channels 必须能被 groups 整除"

        # 1. 预设归一化时间轴 t
        t = torch.linspace(-1, 1, kernel_size)
        self.register_buffer('t', t)

        # 2. 定义可学习参数
        # 基础原子数等于 out_channels，每个原子拥有独立的 a, b, g
        self.w_alpha = nn.Parameter(torch.empty(out_channels, 1))
        self.w_beta = nn.Parameter(torch.empty(out_channels, 1))
        self.w_gamma = nn.Parameter(torch.empty(out_channels, 1))
        # self.w_theta = nn.Parameter(torch.zeros(out_channels, 1))
        # 如果使用 (out_channels, in_channels_per_group,1)：意味着同一个输出通道 $k$，
        # 在面对输入通道 A 时用的是 50Hz 的核，面对输入通道 B 时用的是 60Hz 的核。
        # 此时，$k$ 就不再是一个明确的“原子”，而变成了一个杂乱的多通道组合滤波器
        self.reset_parameters()

    def reset_parameters(self):
        """
        物理启发式的小波参数初始化 (Filter Bank 初始化)
        调用此函数将 w_alpha, w_beta, w_gamma 强制重置为具有物理意义的多尺度小波分布
        """
        # 1. 初始化中心频率 (w_beta) - 使用对数分布覆盖 2.0 到 40.0 的频段
        min_beta = 1
        max_beta = self.length

        log_min = math.log10(min_beta)
        log_max = math.log10(max_beta)

        # 生成对数分布的频率序列，并调整形状为 [out_channels, 1]
        init_beta = torch.logspace(log_min, log_max, self.out_channels).unsqueeze(1)

        # 安全地覆盖现有参数的数据
        with torch.no_grad():
            self.w_beta.copy_(init_beta)

            # 2. 初始化包络宽度 (w_alpha) - 高频高分辨率(窄窗口)，低频低分辨率(宽窗口)
            # 根据 beta 成比例缩放，限制最小值为 2.0 防止窗口完全变平
            init_alpha = torch.clamp(init_beta / 3.0, min=2.0)
            self.w_alpha.copy_(init_alpha)

            # 3. 初始化调频 (w_gamma) - 初始无复杂调频，全填 0
            init_gamma = init_beta/self.out_channels   # 轻微调频，增加模型表达力
            self.w_gamma.copy_(init_gamma)

    def forward(self, x):
        # 3. 生成基础波形 [out_channels, kernel_size]
        # M_k = exp(-a * t^2) * cos(b * t + g * t^2)
        envelope = torch.exp(-torch.abs(self.w_alpha) * (self.t ** 2))
        oscillation = torch.cos(self.w_beta * self.t + self.w_gamma * (self.t ** 2))
        # oscillation = torch.cos(self.w_beta * self.t + self.w_gamma * (self.t ** 2)+self.w_theta)
        kernels = envelope * oscillation

        # 4. L2 归一化 (保持训练稳定)
        kernels = kernels / (torch.norm(kernels, p=2, dim=1, keepdim=True) + 1e-6)
        # 5. 核心逻辑：适配卷积核形状 [out_channels, in_channels/groups, kernel_size]
        # 在 PyTorch 中，分组卷积要求 weight 的形状如上所示
        in_channels_per_group = self.in_channels // self.groups

        # 将 [out_channels, kernel_size] 扩展为 [out_channels, in_channels_per_group, kernel_size]
        wavelet_weight = kernels.unsqueeze(1).repeat(1, in_channels_per_group, 1)

        # 6. 执行卷积
        return F.conv1d(x, wavelet_weight, groups=self.groups, stride=self.stride, padding=self.kernel_size // 2)

    def get_wavelet_kernels(self):
        with torch.no_grad():
            envelope = torch.exp(-torch.abs(self.w_alpha) * (self.t ** 2))
            oscillation = torch.cos(self.w_beta * self.t + self.w_gamma * (self.t ** 2))
            # oscillation = torch.cos(self.w_beta * self.t + self.w_gamma * (self.t ** 2)+self.w_theta)

            kernels = envelope * oscillation
            kernels = kernels / (torch.norm(kernels, p=2, dim=1, keepdim=True) + 1e-6)
            return kernels.cpu().numpy()  # 返回 [out_channels, kernel_size] 的核


class WaveletEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, length=128):
        super().__init__()
        self.waveletConv = WaveletConv(in_dim, out_dim, kernel_size, groups=in_dim, length=length)

        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.waveletConv(x)+x
        x = self.pool(x)
        return x


class Filter(nn.Module):
    def __init__(self, dim, rate):
        super().__init__()
        self.filter = nn.Sequential(
            nn.Conv1d(dim, dim//rate, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim//rate),
            nn.GELU(),
            nn.Conv1d(dim//rate, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.filter(x)


class TFLayer(nn.Module):
    def __init__(self, dim, kernel_size, rate):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )
        self.filter = nn.Sequential(
            nn.Conv1d(dim, dim//rate, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim//rate),
            nn.GELU(),
            nn.Conv1d(dim//rate, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.filter(self.dconv(x))+x
        return x


class WaveletCNN(BaseModel):
    def __init__(self, in_chn=2, length=128, num_classes=10, kernel_size=11, rate=8,
                 dims=[64, 64, 64, 64]):
        super().__init__()
        self.dims = dims

        self.downsample_layers = nn.ModuleList()
        # Stem 层
        stem = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_chn, dims[0], kernel_size=1, bias=False),
            WaveletConv(dims[0], dims[0], kernel_size, groups=dims[0], length=length),

        )

        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            downsample_layer = WaveletEmbedding(dims[i], dims[i + 1], kernel_size, length=length)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            stage = TFLayer(dim=dims[i], kernel_size=kernel_size, rate=rate)
            self.stages.append(stage)

        # 3. 分类头
        self.SE_Block = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 4),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(dims[-1] // 4, dims[-1]),
            nn.Sigmoid()
        )
        self.pred = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # B L C -> B C L
        x = x.transpose(1, 2)
        for i in range(len(self.dims)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.mean([-1])  # 全局平均池化
        x = self.SE_Block(x) * x  # SE 注意力机制
        return self.pred(x)

    def get_wavelet_kernels_featuremap(self, x):
        # 获取第一层 WaveletConv 的核和特征图
        with torch.no_grad():
            wavelet_kernels = self.downsample_layers[0][2].get_wavelet_kernels()  # 获取核
            feature_map = self.downsample_layers[0](x.transpose(1, 2))  # 获取特征图
            return wavelet_kernels, feature_map.cpu().numpy()


# 实例化与测试
if __name__ == '__main__':
    # model = WaveletCNN(in_chn=2, length=128, num_classes=11, kernel_size=11, rate=8,
    #                    dims=[64, 64, 64, 64])
    model = WaveletCNN(in_chn=2, length=1024, num_classes=3, kernel_size=11, rate=8,
                       dims=[64, 64, 64, 64])
    from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (128, 2), as_strings=False, print_per_layer_stat=True, verbose=True)
    macs, params = get_model_complexity_info(model, (1024, 2), as_strings=False,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
