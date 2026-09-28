"""
"Wavelet-based Adaptive Network for Automatic Modulation Recognition under Low SNR"
reproduced by ASME
"""
import torch
import torch.nn as nn
import pywt  # 小波变换库
from pytorch_wavelets import DWT1DForward, DWT1DInverse

from .. import BaseModel

'''
分解层级多一些 levels=5
阈值分通道更优
ECA kernel=3
MCLDNN框架 参数还可以调优  
'''
class WAN(BaseModel):
    def __init__(self, in_len=128, hidden_size=256, wavelet='db8', levels=4, hid_dim=64, num_classes=11):
        super(WAN, self).__init__()
        self.wteb = WTEB(in_len=in_len, hidden_size=hidden_size)
        self.wtrb = WTRB(wavelet=wavelet, levels=levels)
        self.smfb = SMFB(hid_dim=hid_dim, num_classes=num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # b c l
        # 信号恢复路径
        thresholds = self.wteb(x)
        recovered_signal = self.wtrb(x, thresholds)

        # 调制识别路径
        output = self.smfb(recovered_signal)
        return output

    @classmethod
    def from_config(cls, config: dict):
        import inspect
        sig = inspect.signature(cls.__init__)
        # 去掉 self 参数
        params = list(sig.parameters.keys())[1:]

        # 只保留构造函数需要的参数
        filtered = {k: config[k] for k in params if k in config}
        return cls(**filtered)


class WTEB(nn.Module):
    def __init__(self, in_len=128, hidden_size=256):
        super(WTEB, self).__init__()
        self.fc1 = nn.Linear(in_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # 输出阈值

    def forward(self, x):
        '''
        按通道计算阈值
        :param x:
        :return:
        '''
        # x: B C L
        F_t = torch.tanh(self.fc1(x))
        C_t = torch.sigmoid(self.fc2(F_t))  # B C 1
        return C_t


class WTRB(nn.Module):
    def __init__(self, wavelet='db8', levels=3):
        super(WTRB, self).__init__()
        self.wavelet = wavelet
        self.levels = levels

        # 初始化小波变换和逆变换（启用批处理模式）
        self.dwt = DWT1DForward(wave=wavelet, J=levels, mode='zero')
        self.idwt = DWT1DInverse(wave=wavelet, mode='zero')

    def forward(self, x, thresholds):
        """
        输入:
            x: B C L
            thresholds: (batch_size, 1)
        输出:
            recovered_signal: (batch_size, 2, 128)
        """

        # 批量小波分解 -> 返回元组:(LL, [HL, LH, HH...])
        coeffs = self.dwt(x)

        # 应用软阈值（保持系数结构）
        coeffs_thresh = self._batch_apply_threshold(coeffs, thresholds)

        # 批量小波重构
        x_rec = self.idwt(coeffs_thresh)  # B C L

        # 恢复原始形状 (batch, 2, 128)
        return x_rec

    def _batch_apply_threshold(self, coeffs, thresholds):
        '''

        :param coeffs:
        :param thresholds:  [B C 1] or [B 1]
        :return:
        '''

        if thresholds.ndim == 2:
            thresholds = thresholds.unsqueeze(1)
        # 处理近似系数（不应用阈值）
        LL = coeffs[0]

        # 处理细节系数
        Hs = []
        for level_coeffs in coeffs[1]:
            # 软阈值公式: sign(x) * max(|x| - t, 0)
            Hs.append(torch.sign(level_coeffs) *
                      torch.relu(torch.abs(level_coeffs) -
                                 thresholds))

        return (LL, Hs)


class SMFB(nn.Module):
    def __init__(self, hid_dim=64, num_classes=11):
        super(SMFB, self).__init__()
        self.hid_dim = hid_dim
        self.conv1D = nn.Sequential(
            nn.Conv1d(2, hid_dim * 2, kernel_size=8, padding="same", groups=2),
            nn.BatchNorm1d(hid_dim * 2),
            nn.ReLU()
        )
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, kernel_size=(2, 8), padding="same"),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU()
        )
        # ECA模块
        self.eca1 = ECAModule()
        self.eca2 = ECAModule()

        self.conv2D_2 = nn.Sequential(
            nn.Conv2d(1, hid_dim, kernel_size=(2, 8), padding="same"),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU()
        )
        self.conv2D = nn.Sequential(
            nn.Conv2d(hid_dim * 2, hid_dim * 2, kernel_size=(2, 8), padding='valid'),
            nn.BatchNorm2d(hid_dim * 2),
            nn.ReLU()
        )
        # LSTM层
        self.lstm = nn.LSTM(input_size=hid_dim * 2, hidden_size=hid_dim * 2, num_layers=2, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hid_dim * 2, 128),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x形状: (batch_size, 2, 128) [I/Q通道]
        B, C, L = x.shape

        x1 = self.conv1D(x)
        x1 = x1.reshape(B, self.hid_dim, 2, -1)  # (B, hid_dim, 2, L)
        x1 = self.eca1(x1)
        x1 = self.conv2D_1(x1)  # (B, hid_dim, 1, L)

        x2 = self.conv2D_2(x.unsqueeze(1))  # (B, hid_dim, 2, L)
        x = self.eca2(torch.cat([x1, x2], dim=1))  # (B, 2*hid_dim, 2, L)

        x = self.conv2D(x)  # (B, 2*hid_dim, 1, L)
        x = x.squeeze(-2).transpose(1, 2)  # (B, L, 2*hid_dim)

        x, _ = self.lstm(x)  # (B, L, 2*hid_dim)
        x = self.classifier(x[:, -1, :])  # 取最后一个时间步的输出 (B, 2*hid_dim)

        return x


class ECAModule(nn.Module):
    def __init__(self, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x形状: (batch_size, channels, height, width)
        y = self.avg_pool(x)  # (batch_size, channels, 1, 1)
        # print(y.shape)
        y = y.squeeze(-1).transpose(-1, -2)  # (batch_size, 1, channels)
        y = self.conv(y)  # (batch_size, 1, channels)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (batch_size, channels, 1, 1)

        return x * y.expand_as(x)
