"""
FixedWaveletCNN: WaveletCNN 的对比消融模型.

用 ptwt 固定传统小波 CWT (默认 Morlet) 替换可学习小波:
- stem: BN + ptwt.CWT + 通道投影
- 后续下采样: 与 WaveletEmbedding 对应, 残差式 CWT + 池化
- 时频层仍使用相同的 TFLayer
"""
import numpy as np
import pywt
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

import ptwt

from .. import BaseModel
from .WaveletCNN import TFLayer

WAVELET_ALIAS = {
    'morlet': 'morl',
    'morl': 'morl',
    'mexican_hat': 'mexh',
    'mexh': 'mexh',
    'cmor': 'cmor1.5-1.0',
}


def _resolve_wavelet(name):
    key = str(name).lower()
    return WAVELET_ALIAS.get(key, name)


class PtwtCWT(nn.Module):
    """ptwt 连续小波变换封装, 支持 CUDA.

    输入 [B, C, L], 输出 [B, C, S, L] (复小波取模).
    """

    def __init__(self, n_scales, length, wavelet='morl'):
        super().__init__()
        self.n_scales = int(n_scales)
        self.length = int(length)
        self.wavelet = _resolve_wavelet(wavelet)
        scales = np.geomspace(1.0, max(self.length / 4.0, float(self.n_scales)), self.n_scales)
        # ptwt.cwt 内部会 scales.numpy(), 必须放 CPU
        self.scales = np.asarray(scales, dtype=np.float64)

    def forward(self, x):
        B, C, L = x.shape
        coef, _ = ptwt.cwt(x.reshape(B * C, L), self.scales, self.wavelet)  # [S, B*C, L]
        if torch.is_complex(coef):
            coef = coef.abs()
        coef = coef.to(dtype=x.dtype)
        return coef.reshape(self.n_scales, B, C, L).permute(1, 2, 0, 3).contiguous()

    def get_wavelet_kernels(self):
        wav = pywt.ContinuousWavelet(self.wavelet)
        psi, _ = wav.wavefun(level=10)
        return np.asarray(psi)


class CWTStem(nn.Module):
    """传统小波 stem: BN -> ptwt CWT -> 1x1 投影到模型通道."""

    def __init__(self, in_chn, out_dim, length, n_scales, wavelet='morl'):
        super().__init__()
        self.norm = nn.Sequential(
            nn.BatchNorm1d(in_chn),
        ) 
        
        self.cwt = PtwtCWT(n_scales, length, wavelet)
        # self.proj = nn.Sequential(
        #     nn.Conv1d(in_chn * n_scales, out_dim, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(out_dim),
        # )

    def forward(self, x):
        x = self.norm(x)
        x = self.cwt(x).flatten(1, 2)
        print(x.shape)
        return x


class CWTEmbedding(nn.Module):
    """后续层固定 CWT 下采样, 对应 WaveletEmbedding: CWT 残差 + 池化."""

    def __init__(self, in_dim, out_dim, length, n_scales, wavelet='morl'):
        super().__init__()
        # self.cwt = PtwtCWT(n_scales, length, wavelet)
        # self.mix = nn.Conv1d(in_dim * n_scales, in_dim, kernel_size=1, groups=in_dim, bias=False)
        # self.proj = nn.Identity() if in_dim == out_dim else nn.Conv1d(in_dim, out_dim, 1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # y = self.mix(self.cwt(x).flatten(1, 2))
        # return self.pool(self.proj(y + x))
        return self.pool(x)


class FixedWaveletCNN(BaseModel):
    """WaveletCNN 对比模型: ptwt 固定小波 CWT + 时频层."""

    def __init__(self, in_chn=2, length=128, num_classes=10, kernel_size=11, rate=8,
                 dims=None, wavelet='morl', n_scales=8):
        super().__init__()
        if dims is None:
            dims = [64, 64, 64, 64]
        self.dims = list(dims)
        self.n_scales = int(n_scales)
        wavelet = _resolve_wavelet(wavelet)

        self.downsample_layers = nn.ModuleList()
        cur_len = int(length)
        self.downsample_layers.append(
            CWTStem(in_chn, self.dims[0], cur_len, self.n_scales, wavelet=wavelet)
        )
        for i in range(len(self.dims) - 1):
            self.downsample_layers.append(
                CWTEmbedding(self.dims[i], self.dims[i + 1], cur_len, self.n_scales, wavelet=wavelet)
            )
            cur_len = max(cur_len // 2, 1)

        self.stages = nn.ModuleList()
        for i in range(len(self.dims)):
            self.stages.append(TFLayer(dim=self.dims[i], kernel_size=kernel_size, rate=rate))

        self.SE_Block = nn.Sequential(
            nn.Linear(self.dims[-1], self.dims[-1] // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.dims[-1] // 4, self.dims[-1]),
            nn.Sigmoid()
        )
        self.pred = nn.Linear(self.dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
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

        x = x.mean([-1])
        x = self.SE_Block(x) * x
        return self.pred(x)

    def get_wavelet_kernels_featuremap(self, x):
        with torch.no_grad():
            wavelet_kernels = self.downsample_layers[0].cwt.get_wavelet_kernels()
            feature_map = self.downsample_layers[0](x.transpose(1, 2))
            return wavelet_kernels, feature_map.cpu().numpy()


if __name__ == '__main__':
    model = FixedWaveletCNN(in_chn=2, length=128, num_classes=11, kernel_size=11, rate=8,
                           dims=[64, 64, 64, 64],n_scales=32)
    # model = FixedWaveletCNN(in_chn=2, length=1024, num_classes=3, kernel_size=11, rate=8,
    #                     dims=[64, 64, 64, 64],n_scales=32)
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (128, 2), as_strings=False, print_per_layer_stat=True, verbose=True)
    # macs, params = get_model_complexity_info(model, (1024, 2), as_strings=False, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))