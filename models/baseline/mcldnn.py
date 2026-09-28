"""
"A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition"
https://ieeexplore.ieee.org/abstract/document/9106397

reproduced by IVSN
"""


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Sequential
from .. import BaseModel

# num_classes = 11
# BasicBlock {{{
class Mcldnn(BaseModel):
    """`MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_ backbone
    The input for LSTM is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=11):
        super(Mcldnn, self).__init__()
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(2, 8), padding='same', ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding='valid'),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, num_layers=2)
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.SELU(True),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.SELU(True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x = x.transpose(1, 2)  # B C L
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.conv2(x[:, :, 0, :])
        x3 = self.conv3(x[:, :, 1, :])
        x4 = self.conv4(torch.stack([x2, x3], dim=2))
        x5 = self.conv5(torch.cat([x1, x4], dim=1))  # B C 1 L
        x = x5.squeeze(2).transpose(1, 2)  # B L C
        x, _ = self.lstm(x)
        x = self.classifier(x[:, -1, :])

        return x




class MCLDNN(nn.Module):

    def __init__(self, num_classes):
        super(MCLDNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=50,
            kernel_size=7,
            bias=False,
            padding=3,
        )
        self.conv2 = Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=100,
                kernel_size=7,
                bias=False,
                padding=3,
                groups=2
            ),
            nn.ReLU(True),
            nn.Conv1d(
                in_channels=100,
                out_channels=50,
                kernel_size=7,
                bias=False,
                padding=3,
            ))
        self.conv3 = nn.Conv1d(
            in_channels=100,
            out_channels=100,
            kernel_size=5,
            bias=False
        )
        self.lstm1 = nn.LSTM(
            input_size=100,
            hidden_size=128,
            num_layers=1,
            bias=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            bias=False,
            batch_first=True
        )
        self.fc = Sequential(
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        assert len(x.shape) == 3 and x.shape[1] == 2
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = F.relu(torch.cat([x1, x2], dim=1))
        x3 = F.relu(self.conv3(x3))
        # print(x3.shape)
        x3, _ = self.lstm1(x3.transpose(2, 1))
        _, (x3, __) = self.lstm2(x3)
        # print(x3.shape)
        x3 = self.fc(x3.squeeze())

        return x3

    @classmethod
    def from_config(cls, config: dict):
        import inspect
        sig = inspect.signature(cls.__init__)
        # 去掉 self 参数
        params = list(sig.parameters.keys())[1:]

        # 只保留构造函数需要的参数
        filtered = {k: config[k] for k in params if k in config}
        return cls(**filtered)


class MCLDNN_r(nn.Module):

    def __init__(self, config):
        super(MCLDNN_r, self).__init__()
        self.config = config

        self.conv1 = nn.Conv2d(1, 50, kernel_size=(2, 8), padding=(0, 4), bias=False)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8, bias=False)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8, bias=False)
        self.conv4 = nn.Conv2d(50, 50, kernel_size=(2, 8), bias=False)
        self.conv5 = nn.Conv2d(100, 100, kernel_size=(2, 5), bias=False)
        self.lstm1 = nn.LSTM(
            input_size=100,
            hidden_size=128,
            num_layers=1,
            bias=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            bias=False,
            batch_first=True
        )
        self.fc = Sequential(
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.Dropout(0.5),
            nn.Linear(128, self.config['num_classes'])
        )

    def forward(self, x):
        # [B,L,C]
        x = x.transpose(1, 2)
        assert len(x.shape) == 3 and x.shape[1] == 2
        x = torch.unsqueeze(x, 1)  # [B,1,C=2,L=128]
        x1 = F.relu(self.conv1(x))  # [B,50,1,121]
        x2 = F.relu(self.conv2(x[:, :, 0, :]))  # [B,1,128]->[B,50,121]
        x3 = F.relu(self.conv3(x[:, :, 1, :]))  # [B,1,128]->[B,50,121]
        concatenate1 = torch.cat([x2.unsqueeze(dim=2), x3.unsqueeze(dim=2)], dim=2)  # [B,50,2,121]
        x4 = F.relu(self.conv4(concatenate1))  # [B,50,2,121]->[B,50,1,114]
        print(x1.shape)
        print(x4.shape)
        concatenate2 = torch.cat([x1, x4], dim=1)
        x5 = F.relu(self.conv5(concatenate2))  # [N,50,128]
        x, _ = self.lstm1(x5.transpose(2, 1))
        _, (x, __) = self.lstm2(x)
        x = self.fc(x.squeeze())

        return x3
