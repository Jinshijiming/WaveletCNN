import numpy as np
import random

from torchvision.transforms import Compose,Lambda

import torch


class AddNoise:
    """
    添加高斯噪声。
    """

    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, signal):
        noise = self.noise_level * torch.randn_like(signal)
        return signal + noise


class TimeShift:
    """
    时间偏移。
    """

    def __call__(self, signal):
        max_len = signal.shape[-1]
        shift = torch.randint(-max_len, max_len, (1,)).item()
        return torch.roll(signal, shifts=shift, dims=1)


class Scaling:
    """
    幅度缩放。
    """

    def __init__(self, scale_min=0.8, scale_max=1.2):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, signal):
        scale = torch.empty(1).uniform_(self.scale_min, self.scale_max)
        return signal * scale


class RandomDropout:
    """
    随机丢弃。
    """

    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, signal):
        drop_mask = torch.rand_like(signal) > self.drop_prob
        return signal * drop_mask


class GivenTransformation:
    """
    给定变换：对 IQ 序列进行随机旋转。
    """

    def __call__(self, x_n):
        alpha = torch.tensor(random.choice([0, 0.5, 1, 1.5])) * torch.pi
        rotation_matrix = torch.tensor([
            [torch.cos(alpha), -torch.sin(alpha)],
            [torch.sin(alpha), torch.cos(alpha)]
        ], dtype=x_n.dtype)
        return torch.matmul(rotation_matrix, x_n)


class AmplitudeReversal:
    """
    幅度反转。
    """

    def __call__(self, x_n):
        i = torch.randint(0, 2, (1,))
        j = torch.randint(0, 2, (1,))
        return torch.stack([(-1) ** i * x_n[0], (-1) ** j * x_n[1]], dim=0)


class TimeReversal:
    """
    时间反转，加入随机性。
    """

    def __init__(self, p=0.5):
        """
        :param p: 进行时间反转的概率，默认为 0.5。
        """
        self.p = p

    def __call__(self, x_n):
        """
        根据概率 p 决定是否对信号进行时间反转。
        :param x_n: 输入的信号，形状为 (2, n)。
        :return: 时间反转后的信号（如果随机选择进行反转）。
        """
        if torch.rand(1).item() < self.p:
            return x_n.flip(dims=[0])
        else:
            return x_n


class LocalRandomTransform:
    """
    对信号的局部区域进行随机变换。
    """

    def __init__(self, transform, min_ratio=0.1, max_ratio=0.5):
        """
        :param transform: 要应用的变换操作（如 AddNoise、TimeShift 等）。
        :param min_ratio: 局部区域的最小比例（相对于信号长度）。
        :param max_ratio: 局部区域的最大比例（相对于信号长度）。
        """
        self.transform = transform
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, signal):
        """
        对信号的局部区域进行随机变换。
        :param signal: 输入信号，形状为 (2, n)。
        :return: 变换后的信号。
        """
        n = signal.shape[1]  # 信号长度
        ratio = torch.empty(1).uniform_(self.min_ratio, self.max_ratio).item()
        length = int(n * ratio)
        if n - length <= 0:
            start = 0
        else:
            start = torch.randint(0, n - length, (1,)).item()
        local_signal = signal[:, start:start + length]
        transformed_local_signal = self.transform(local_signal)
        signal[:, start:start + length] = transformed_local_signal
        return signal


class ToTensor:
    """
    将输入转换为 PyTorch 张量。
    """

    def __call__(self, x_n):
        return torch.tensor(x_n, dtype=torch.float32)


class Normalize:
    """
    对输入进行归一化处理。
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x_n):
        self.mean = torch.as_tensor(self.mean)
        self.std = torch.as_tensor(self.std)
        if self.mean.ndim < 2:
            self.mean = self.mean.view(-1, 1)
        if self.std.ndim < 2:
            self.std = self.std.view(-1, 1)
        return (x_n - self.mean) / self.std


class MaxMinNormalization:
    """
    对输入进行最大最小归一化处理。
    """

    def __init__(self, min_val=-1, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x_n: torch.Tensor):
        # 针对每个通道分别做最大最小归一化
        x_min, _ = x_n.min(dim=-1, keepdim=True)
        x_max, _ = x_n.max(dim=-1, keepdim=True)
        return (x_n - x_min) / (x_max - x_min) * (self.max_val - self.min_val) + self.min_val

class UnitPowerComplexNorm:
    """
    复数（IQ联合）单位功率归一化：I 和 Q 共享一个归一化因子。
    输入形状: [N, 2, L] 或 [2, L]
    """

    def __call__(self, x: torch.Tensor):
        # 确保输入通道为2（I,Q）
        assert x.size(-2) == 2, "IQ 通道必须为2个 [I, Q]"

        # 功率 = I^2 + Q^2
        power = x.pow(2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)  # shape [..., 1, 1]

        # 避免除零
        norm = torch.sqrt(power + 1e-8)

        # I,Q 同比例缩放
        x_norm = x / norm

        return x_norm
class Identity:
    def __call__(self, x):
        return x
def normlization(mean, std, min=-1, max=1, norm='mean-std'):
    if norm == 'mean-std':
        return Normalize(mean, std)

    elif norm == 'max-min':
        return MaxMinNormalization(min, max)
    elif norm == 'unit-power':
        return UnitPowerComplexNorm()
    else:
        return Identity()


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        MaxMinNormalization()
    ])
    # 示例 IQ 序列
    I = np.array([1, 2, 3, 4, 5])
    Q = np.array([5, 4, 3, 2, 1])
    x_n = np.array([I, Q])
    print("原始 IQ 序列:")
    print(x_n)

    # 测试 add_noise
    x_n_noise = transform(x_n)
    print("\n添加噪声后的 IQ 序列:")
    print(x_n_noise)
