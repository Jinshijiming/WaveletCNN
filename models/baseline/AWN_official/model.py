"""
part of codes is adopted from:
https://github.com/mxbastidasr/DAWN_WACV2020
"""

import torch
import torch.nn as nn

from .lifting import LiftingScheme
from ...BaseModel import BaseModel


class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, kernel_size, regu_details, regu_approx):
        super(LevelTWaveNet, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.wavelet = LiftingScheme(in_planes, kernel_size=kernel_size)

    def forward(self, x):
        """
        Conduct decomposition and calculate regularization terms
        :param x:
        :return: approx component, details component, regularization terms
        """
        global regu_d, regu_c
        (L, H) = self.wavelet(x)  # 10 9 128
        approx = L
        details = H
        if self.regu_approx + self.regu_details != 0.0:
            if self.regu_details:
                regu_d = self.regu_details * H.abs().mean()
            # Constrain on the approximation
            if self.regu_approx:
                regu_c = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)

            if self.regu_approx == 0.0:
                # Only the details
                regu = regu_d
            elif self.regu_details == 0.0:
                # Only the approximation
                regu = regu_c
            else:
                # Both
                regu = regu_d + regu_c

            return approx, details, regu


class AWN(BaseModel):
    def __init__(self,
                 num_classes,
                 num_levels=1,
                 in_channels=64,
                 kernel_size=3,
                 latent_dim=256,
                 regu_details=0.01,
                 regu_approx=0.01):
        super(AWN, self).__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = self.in_channels * (self.num_levels + 1)
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.regu_details = regu_details
        self.regu_approx = regu_approx

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            # Call a 2d Conv to integrate I, Q channels
            nn.Conv2d(1, self.in_channels,
                      kernel_size=(2, 7), stride=(1,), bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels,
                      kernel_size=(5,), stride=(1,), padding=(2,), bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.levels = nn.ModuleList()

        for i in range(self.num_levels):
            self.levels.add_module(
                'level_' + str(i),
                LevelTWaveNet(self.in_channels,
                              self.kernel_size,
                              self.regu_details,
                              self.regu_approx)
            )

        self.SE_attention_score = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels // 4, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels // 4, self.out_channels, bias=False),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(self.out_channels, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.latent_dim, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # x:[N, 2, T] -> [N, 1, 2, T]
        x = self.conv1(x)
        x = x.squeeze(2)  # x:[N, C, 1, T] -> [N, C, T]
        x = self.conv2(x)
        regu_sum = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details

        for l in self.levels:
            x, details, regu = l(x)
            regu_sum += [regu]
            det += [self.avgpool(details)]
        aprox = self.avgpool(x)
        det += [aprox]

        x = torch.cat(det, 1)
        x = x.view(-1, x.size()[1])
        x = torch.mul(self.SE_attention_score(x), x)

        logit = self.fc(x)

        return logit, regu_sum

    def train_step(self, batch: torch.Tensor):
        x, y = batch
        x = x.to(self.device)
        y = y[:, 1].to(self.device)
        logits, regu_sum = self(x)
        loss = self.criterion(logits, y) + sum(regu_sum)
        # loss = self.criterion(logits, y)
        return {
            'loss': loss,
            'logits': logits,
            'pred': logits.argmax(dim=1),
            'target': y
        }

if __name__ == '__main__':
    model = AWN(num_classes=10, num_levels=1, in_channels=64, kernel_size=3, latent_dim=256, regu_details=0.01, regu_approx=0.01)
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (128, 2), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    from thop import profile
    import json
    input = torch.randn(1, 128,2 )
    flops, params,layers = profile(model, inputs=(input,),ret_layer_info=True)
    print('flops: ', flops, 'params: ', params)
    print(json.dumps(layers, indent=4))