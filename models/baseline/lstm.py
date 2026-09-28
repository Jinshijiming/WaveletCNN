"""
"Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors"
https://ieeexplore.ieee.org/abstract/document/8357902

reproduced by IVSN
"""
import inspect
from typing import Dict

import torch
import torch.nn as nn
from .. import BaseModel


class lstm2(BaseModel):
    def __init__(self, num_classes, length, num_channel):
        nn.Module.__init__(self)

        self.lstm1 = nn.LSTM(
            input_size=num_channel,
            hidden_size=128,
            num_layers=1,
            bias=False,
            batch_first=True
        )
        self.bn1 = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            bias=False,
            batch_first=True
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(length * 64, num_classes)

    def forward(self, x):
        '''
        [B,L,F]
        '''

        x, _ = self.lstm1(x)
        x = self.bn1(x.transpose(2, 1)).transpose(2, 1)
        x, _ = self.lstm2(x)
        x = self.bn2(x.transpose(2, 1)).transpose(2, 1)
        x = torch.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        return x




if __name__ == '__main__':
    data = torch.randn(20, 512, 2)
    model = lstm2(11, 512, 2)
    print(model(data).shape)
