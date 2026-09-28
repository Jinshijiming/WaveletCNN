"""
"A Transformer-based CTDNN Structure for Automatic Modulation Recognition"
https://ieeexplore.ieee.org/abstract/document/9674558

reproduced by ASME
"""

import torch
import torch.nn as nn
import numpy as np

from .. import BaseModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# -----------------------------
# CTDNN Model
# -----------------------------
class CTDNN(BaseModel):
    def __init__(self, in_len=128, in_chn=2, num_kernel=256, num_heads=2, num_layers=2, ff_dim=512, num_classes=10):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(1, num_kernel, kernel_size=(in_chn, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_kernel),
            nn.Tanh()
        )
        self.flatten = nn.Flatten(1)  # [B, C, 128]
        self.pe = PositionalEncoding(d_model=num_kernel, max_len=in_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_kernel, nhead=num_heads, dim_feedforward=ff_dim,
                                                   dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(num_kernel, num_kernel * 2),
            nn.ReLU(),
            nn.Linear(num_kernel * 2, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 2, 128]
        x = self.embedding(x)  # [B, C, 1, 128]
        x = x.squeeze(2)  # [B, C, 128]
        x = x.permute(0, 2, 1)  # [B, 128, C]
        x = self.pe(x)  # Add positional encoding
        x = self.transformer(x)  # [B, 128, C]

        # x = x[:, 0, :]
        x = x.mean(dim=1)

        return self.classifier(x)

