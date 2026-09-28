"""
MCformer: A Transformer Based Deep Neural Network for Automatic Modulation Classification
https://ieeexplore.ieee.org/abstract/document/9685815

reproduced by ASME
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .. import BaseModel



class SelfAttentionTransformerBERT(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob=0.1):
        super(SelfAttentionTransformerBERT, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_prob)

        # Feed-forward neural network (FFN)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_prob)
        )

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Multi-head self-attention
        x = x.permute(2, 0, 1)  # Shape: [R, batch_size, C]
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward neural network
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x.permute(1, 2, 0)  # Shape: [batch_size, C, R]


class MCformer(BaseModel):
    def __init__(self, hidden_size=32, kernel_size=65, d_ff=256, num_heads=4, num_classes=11):
        super(MCformer, self).__init__()

        # Convolutional layer with SELU activation
        self.conv_layer = nn.Conv2d(2, hidden_size, kernel_size=(kernel_size, 1), padding='same')
        self.selu = nn.SELU()

        # # Transformer Encoder layers
        # self.transformer_layers = nn.ModuleList([
        #     SelfAttentionTransformerBERT(hidden_size, num_heads) for _ in range(4)  # 4 self-attention layers
        # ])

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=d_ff, batch_first=True),
            num_layers=4
        )
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.fc1 = nn.Linear(4 * hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B C L]
        x = x.unsqueeze(-1)  # unsqueeze the last dimension
        # Convolutional layer
        enc_out = self.conv_layer(x)
        enc_out = self.selu(enc_out)

        # Reshape
        # x = x.view(-1, 128, 32)
        enc_out = enc_out.squeeze(-1)

        # # Transformer Encoder layers
        # for layer in self.transformer_layers:
        #     x = layer(x)
        # x = x[:, :, :4]

        enc_out = enc_out.permute(0, 2, 1)
        x = self.transformer(enc_out)
        x = x[:, :4, :]
        x = self.flatten(x)
        # Fully connected layers
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)

        return x


