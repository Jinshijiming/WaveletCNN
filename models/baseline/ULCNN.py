"""
"Ultralight Convolutional Neural Network for Automatic Modulation Classification in Internet of Unmanned Aerial Vehicles"
reproduced by ASME

https://github.com/BeechburgPieStar/ULCNN
"""

from .complexLayers import CConv1d, CBatchNorm1d, CReLU
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import BaseModel
from calflops import calculate_flops
import math


class ComplexConv1D(nn.Module):
    """Complex-valued 1D Convolution Layer"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Real and imaginary parts of the weights
        self.weight_real = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.weight_imag = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Xavier/Glorot initialization for complex weights
        fan_in = self.in_channels * self.kernel_size
        fan_out = self.out_channels * self.kernel_size
        bound = math.sqrt(2.0 / (fan_in + fan_out))

        nn.init.uniform_(self.weight_real, -bound, bound)
        nn.init.uniform_(self.weight_imag, -bound, bound)

    def forward(self, x):
        # x is expected to be of shape (batch, length, 2) where last dim is [real, imag]
        # Convert to (batch, 2, length) for conv1d
        x = x.permute(0, 2, 1)
        x_real = x[:, 0:1, :]  # (batch, 1, length)
        x_imag = x[:, 1:2, :]  # (batch, 1, length)

        # Complex convolution: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        conv_real = F.conv1d(x_real, self.weight_real, stride=self.stride, padding=self.padding) - \
                    F.conv1d(x_imag, self.weight_imag, stride=self.stride, padding=self.padding)
        conv_imag = F.conv1d(x_real, self.weight_imag, stride=self.stride, padding=self.padding) + \
                    F.conv1d(x_imag, self.weight_real, stride=self.stride, padding=self.padding)

        # Concatenate and permute back to (batch, length, channels)
        output = torch.cat([conv_real, conv_imag], dim=1)  # (batch, 2*out_channels, length)
        output = output.permute(0, 2, 1)  # (batch, length, 2*out_channels)

        return output


class ComplexBatchNorm1D(nn.Module):
    """Complex-valued Batch Normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(ComplexBatchNorm1D, self).__init__()
        self.num_features = num_features // 2  # Since we have real and imaginary parts
        self.eps = eps
        self.momentum = momentum

        # Parameters for real and imaginary parts
        self.weight_real = nn.Parameter(torch.ones(self.num_features))
        self.weight_imag = nn.Parameter(torch.ones(self.num_features))
        self.bias_real = nn.Parameter(torch.zeros(self.num_features))
        self.bias_imag = nn.Parameter(torch.zeros(self.num_features))

        # Running statistics
        self.register_buffer('running_mean_real', torch.zeros(self.num_features))
        self.register_buffer('running_mean_imag', torch.zeros(self.num_features))
        self.register_buffer('running_var_real', torch.ones(self.num_features))
        self.register_buffer('running_var_imag', torch.ones(self.num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x shape: (batch, length, 2*num_features)
        batch_size, length, channels = x.shape

        # Split real and imaginary parts
        x_real = x[:, :, :self.num_features]
        x_imag = x[:, :, self.num_features:]

        if self.training:
            # Calculate batch statistics
            mean_real = x_real.mean(dim=[0, 1])
            mean_imag = x_imag.mean(dim=[0, 1])
            var_real = x_real.var(dim=[0, 1], unbiased=False)
            var_imag = x_imag.var(dim=[0, 1], unbiased=False)

            # Update running statistics
            self.running_mean_real = (1 - self.momentum) * self.running_mean_real + self.momentum * mean_real
            self.running_mean_imag = (1 - self.momentum) * self.running_mean_imag + self.momentum * mean_imag
            self.running_var_real = (1 - self.momentum) * self.running_var_real + self.momentum * var_real
            self.running_var_imag = (1 - self.momentum) * self.running_var_imag + self.momentum * var_imag

            mean_r, mean_i = mean_real, mean_imag
            var_r, var_i = var_real, var_imag
        else:
            mean_r, mean_i = self.running_mean_real, self.running_mean_imag
            var_r, var_i = self.running_var_real, self.running_var_imag

        # Normalize
        x_real_norm = (x_real - mean_r) / torch.sqrt(var_r + self.eps)
        x_imag_norm = (x_imag - mean_i) / torch.sqrt(var_i + self.eps)

        # Apply scale and shift
        x_real_out = self.weight_real * x_real_norm + self.bias_real
        x_imag_out = self.weight_imag * x_imag_norm + self.bias_imag

        return torch.cat([x_real_out, x_imag_out], dim=-1)


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, L = x.size()
        assert C % self.groups == 0, "通道数必须能被 groups 整除"

        x = x.view(N, self.groups, C // self.groups, L)
        x = x.transpose(1, 2).contiguous()
        x = x.view(N, C, L)
        return x


class CA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''

        :param x: B C L
        :return:
        '''
        avg_out = self.fc(self.avg_pool(x))  # B C 1 -> B C//r 1 -> B C 1
        max_out = self.fc(self.max_pool(x))  # B C 1
        out = avg_out + max_out
        return x * self.sigmoid(out)


class ULCNN(BaseModel):
    def __init__(self, num_classes=11):
        super().__init__()

        self.iqcf = nn.Sequential(
            ComplexConv1D(1, 16, kernel_size=5, stride=1, padding='same'),
            ComplexBatchNorm1D(32),
            # CReLU(),
        )
        self.fmdr = nn.ModuleList()
        for _ in range(6):
            self.fmdr.append(
                nn.Sequential(
                    nn.Conv1d(32, 32, kernel_size=5, stride=2, groups=32, padding=2),
                    nn.Conv1d(32, 32, kernel_size=1, stride=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    ChannelShuffle(2),
                    CA(32, 16)
                )
            )
        self.classifier = nn.Sequential(
            nn.Linear(32, num_classes),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x = torch.complex(x[:, :, 0], x[:, :, 1]).unsqueeze(1)  # B K C-> B K-> B 1 K
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.iqcf(x)
        # x = torch.cat((x.real, x.imag), dim=1)
        x = F.relu(x).permute(0,2,1)
        x1 = self.fmdr[0](x)
        x2 = self.fmdr[1](x1)
        x3 = self.fmdr[2](x2)
        x4 = self.fmdr[3](x3)
        x5 = self.fmdr[4](x4)
        x6 = self.fmdr[5](x5)
        x = x4.mean(dim=-1) + x5.mean(dim=-1) + x6.mean(dim=-1)
        return self.classifier(x)


if __name__ == '__main__':
    model = ULCNN(num_classes=11)
    x = torch.randn(2, 128, 2)
    y = model(x)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (128, 2), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))