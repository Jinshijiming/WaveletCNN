"""
"Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition"
https://ieeexplore.ieee.org/abstract/document/10531070

transferred by asme
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from einops.layers.torch import Rearrange
from torchvision import models


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv1d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv1d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv1d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size, expand_ratio=2):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        )
        self.v = nn.Conv1d(dim, dim, 1)
        self.proj = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.att(x) * self.v(x)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, index, dim, kernel_size, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1[:, None] * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2[:, None] * self.mlp(x))
        return x


class Conv2Former(nn.Module):
    def __init__(self, in_chn=2, num_classes=1000, kernel_size=7, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            Rearrange('b c l ->b 1 c l'),
            nn.Conv2d(1, dims[0] // 2, kernel_size=(in_chn, kernel_size), bias=False),
            Rearrange('b c 1 l -> b c l'),
            nn.BatchNorm1d(dims[0] // 2),
            nn.GELU(),

            nn.Conv1d(dims[0] // 2, dims[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(dims[0] // 2),
            nn.GELU(),

            nn.Conv1d(dims[0] // 2, dims[0], kernel_size=2, stride=2, bias=False),
        )

        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            stride = 2
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                # nn.Conv1d(dims[i], dims[i+1], 1),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=stride, stride=stride),  # TODO: 改进池化部分
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(index=cur + j, dim=dims[i], kernel_size=kernel_size, drop_path=dp_rates[cur + j],
                        mlp_ratio=mlp_ratios[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.head = nn.Sequential(
            nn.Conv1d(dims[-1], dims[-1] * 4, 1),
            nn.GELU(),
            LayerNorm(dims[-1] * 4, eps=1e-6, data_format="channels_first")
        )
        self.pred = nn.Linear(dims[-1] * 4, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(len(self.dims)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)
        return x.mean([-1])  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pred(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, length, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, length).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


def conv2former_n(kernel_size=11, **kwargs):
    model = Conv2Former(kernel_size=kernel_size, dims=[64, 64, 64, 64], mlp_ratios=[4, 4, 4, 4], depths=[1, 1, 1, 1],
                        **kwargs)
    return model


def conv2former_t(kernel_size=11, **kwargs):
    model = Conv2Former(kernel_size=kernel_size, dims=[72, 144, 288, 576], mlp_ratios=[
        4, 4, 4, 4], depths=[3, 3, 12, 3], **kwargs)
    return model


def conv2former_s(kernel_size=11, **kwargs):
    model = Conv2Former(kernel_size=kernel_size, dims=[72, 144, 288, 576], mlp_ratios=[
        4, 4, 4, 4], depths=[4, 4, 32, 4], **kwargs)
    return model


def conv2former_b(kernel_size=11, **kwargs):
    model = Conv2Former(kernel_size=kernel_size, dims=[96, 192, 384, 768], mlp_ratios=[
        4, 4, 4, 4], depths=[4, 4, 34, 4], **kwargs)
    return model


def conv2former_b_22k(kernel_size=11, **kwargs):
    model = Conv2Former(kernel_size=kernel_size, dims=[96, 192, 384, 768], mlp_ratios=[
        4, 4, 4, 4], depths=[4, 4, 34, 4], **kwargs)
    return model


def conv2former_l(kernel_size=11, **kwargs):
    model = Conv2Former(kernel_size=kernel_size, dims=[128, 256, 512, 1024], mlp_ratios=[
        4, 4, 4, 4], depths=[4, 4, 48, 4], **kwargs)
    return model


if __name__ == '__main__':
    model = conv2former_b(num_classes=1000)
    input = torch.randn(1, 3, 224)
    output = model(input)
    print(output.shape)
