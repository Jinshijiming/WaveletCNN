"""
"Robust and Efficient Modulation Recognition with Pyramid Signal Transformer"
reproduced by ASME
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .. import BaseModel
# ------------------------------
# Segment Embedding
# ------------------------------
class SegmentEmbedding(nn.Module):
    def __init__(self, in_ch, out_ch, seg_len, stride):
        super().__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.linear = nn.Linear(in_ch * seg_len, out_ch)
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):  # x: [B, C, L]
        B, C, L = x.shape
        unfold = x.unfold(dimension=2, size=self.seg_len, step=self.stride)  # [B, C, L, seg_len]
        L = unfold.shape[2]
        x_seg = unfold.permute(0, 2, 1, 3).reshape(B, L, C * self.seg_len)  # [B, L, C*seg_len]
        x_seg = self.linear(x_seg).permute(0, 2, 1)  # [B, out_ch, L]
        return self.norm(x_seg)


class MultiHeadDualAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.dw_conv = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=3, padding='same', groups=self.head_dim)

    def forward(self, x):  # x: [B, L, C]
        B, L, C = x.shape
        qkv = self.qkv_proj(x)  # [B, L, 3C]
        qkv = qkv.view(B, L, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # each: [B, H, L, D]

        # Linear Self-Attention with einsum
        k_softmax = F.softmax(k, dim=-1)  # [B, H, L, D]
        lsa = torch.einsum('bhlc,bhld->bhcd', q, k_softmax) / math.sqrt(L)  # [B, H, C, D]
        lsa_out = torch.einsum('bhcd,bhsd->bshc', lsa, v).reshape(B, L, C)

        # Contextual Scaling Attention with einsum
        q_reshape = rearrange(q, 'b h l d -> (b h) d l')  # [B*H, D, L]
        csa_weight = torch.sigmoid(self.dw_conv(q_reshape)).reshape(B, -1, self.head_dim, L)  # [B,H,D,L]
        csa_out = torch.einsum('bhdl,bhld->blhd', csa_weight, v).reshape(B, L, C)

        return (lsa_out + csa_out).transpose(1, 2)  # [B, C, L]


# ------------------------------
# LFF Block
# ------------------------------
class LocalFeedForward(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        hidden = d_model * expansion
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, hidden, 1),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding='same', groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, 1)
        )

    def forward(self, x):  # x: [B, C, L]
        return self.ff(x)


# ------------------------------
# Transformer Block with DA + LFF
# ------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(d_model)
        self.attn = MultiHeadDualAttention(d_model, num_heads)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.ff = LocalFeedForward(d_model)

    def forward(self, x):  # x: [B, C, L]
        x = self.attn(self.norm1(x).transpose(1, 2)) + x
        x = self.ff(self.norm2(x)) + x
        return x


# ------------------------------
# SigFormer Model
# ------------------------------
class SigFormer(BaseModel):
    def __init__(self, in_len, num_heads=4, in_chn=2, channels=[8, 16, 32, 64], seg_lens=[4,4,7,7],
                 strides=[2, 2, 2, 2], num_classes=10):
        super().__init__()
        self.stages = nn.ModuleList()
        for i in range(4):
            seg = SegmentEmbedding(in_chn, channels[i], seg_lens[i], strides[i])
            blk = TransformerBlock(channels[i], num_heads=num_heads)
            self.stages.append(nn.Sequential(seg, blk))


            # blk = nn.TransformerEncoderLayer(channels[i], num_heads, 4 * channels[i], batch_first=True, norm_first=True)
            # self.stages.append(nn.Sequential(seg,
            #                                  Rearrange('b c l -> b l c'),
            #                                  blk,
            #                                  Rearrange('b l c -> b c l')
            #                                  ))
            in_chn = channels[i]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], channels[-1]*2),
            nn.GELU(),
            nn.Linear(channels[-1]*2, num_classes)
        )

    def forward(self, x):  # x: [B, L,C]
        x = x.permute(0, 2, 1)
        for stage in self.stages:
            x = stage(x)  # [B, C, L]
        return self.classifier(x)



# ------------------------------
# Label Smoothing Loss
# ------------------------------