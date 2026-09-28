"""
"SigMixer:基于局部信息的电磁调制分类模型"

official implementation
"""
import torch
import torch.nn as nn
from .. import BaseModel


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Stem(nn.Module):
    def __init__(self, in_chans, dim, patch_size, activation=nn.GELU):
        super().__init__()
        self.stem = nn.Sequential(  # patch embedding
            nn.Conv1d(in_chans, dim, kernel_size=patch_size, stride=patch_size, bias=False),
            activation(),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return self.stem(x)


class Block(nn.Module):
    def __init__(self, dim, kernel_size, activation, depth):
        super().__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size // 2, bias=False),
                    activation(),
                    nn.BatchNorm1d(dim),
                )),
                nn.Conv1d(dim, dim, kernel_size=1, bias=False),
                activation(),
                nn.BatchNorm1d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        return self.block(x)


class ConvMixer(BaseModel):  # 最佳
    def __init__(self, dim=[256, 256, 256, 768], depth=5, kernel_size=[7, 7, 7, 7], patch_size=4, in_chans=[1, 1, 2],
                 num_classes=11,
                 activation=nn.GELU):
        super().__init__()
        self.stems = nn.ModuleList()
        for i in range(len(in_chans)):
            stem = Stem(in_chans[i], dim[i], patch_size, activation=nn.SELU)
            self.stems.append(stem)

        self.blocks = nn.ModuleList()
        for i in range(len(dim)):
            block = Block(dim[i], kernel_size[i], activation, depth)
            self.blocks.append(block)

        self.lstm1 = nn.LSTM(input_size=dim[-1], hidden_size=128, num_layers=1, bias=False, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, bias=False, batch_first=True)
        self.classfier = nn.Linear(128, num_classes)
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        # [B,L,N]
        # aml,phase,fre=cal_amplitude_phase_fre(x)
        # x=torch.cat([aml,phase],dim=-1)

        x = x.transpose(2, 1)
        x0 = x[:, 0].unsqueeze(dim=1)
        x1 = x[:, 1].unsqueeze(dim=1)

        x0 = self.stems[0](x0)
        x0 = self.blocks[0](x0)

        x1 = self.stems[1](x1)
        x1 = self.blocks[1](x1)

        x2 = self.stems[2](x)
        x2 = self.blocks[2](x2)

        x = torch.cat((x0, x1, x2), 1)
        x = self.blocks[3](x)

        x = self.pooling(x)
        # print(x.shape)
        x, _ = self.lstm1(x.unsqueeze(dim=1))
        _, (x, __) = self.lstm2(x)

        x = self.classfier(x.squeeze())
        return x


if __name__ == '__main__':
    B = 128
    N = 2
    L = 128
    data = torch.randn([B, L, N])
    model = ConvMixer(patch_size=4, kernel_size=[7, 7, 7, 7], depth=5, num_classes=10,
                      in_chans=[1, 1, 2], dim=[256, 256, 256, 256 * 3])

    out = model(data)
