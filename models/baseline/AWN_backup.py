"""
"Multilevel Adaptive Wavelet Decomposition Network Based Automatic Modulation Recognition: Exploiting Time-Frequency Multiscale Correlations"
https://ieeexplore.ieee.org/abstract/document/10856202

reproduced by asme
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import BaseModel
from torchvision import models

class Operator(nn.Module):
    def __init__(self, in_planes, kernel_size=3, dropout=0.):
        super(Operator, self).__init__()

        pad = (kernel_size - 1) // 2 + 1

        self.operator = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Operator as Predictor() or Updator()
        :param x:
        :return: P(x) or U(x)
        """
        x = self.operator(x)
        return x


class LiftingScheme1D(nn.Module):
    def __init__(self, hid_chn, levels=2):
        super().__init__()
        self.levels = levels

        self.predicts = nn.ModuleList()
        self.updates = nn.ModuleList()
        for _ in range(levels):
            self.predicts.append(
                Operator(hid_chn, 3)
            )
            self.updates.append(
                Operator(hid_chn, 3)
            )

    def forward(self, x):
        '''

        :param x: B C L
        :return:
        '''
        x_even = x[:, :, ::2]
        x_odd = x[:, :, 1::2]
        c_list = [x]
        d_list = []
        for i in range(self.levels):
            c = x_even + self.updates[i](x_odd)
            d = x_odd - self.predicts[i](c)

            x_even = c[:, :, ::2]
            x_odd = c[:, :, 1::2]
            c_list.append(c)
            d_list.append(d)
        return c_list, d_list


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.fc(x).view(b, c)
        return torch.mul(x, y)


class AWN(BaseModel):
    def __init__(self, in_chn, hid_chn=64, levels=3, num_classes=11, lambda1=0.01, lambda2=0.01):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, hid_chn, kernel_size=(2, 7), bias=False),
            nn.BatchNorm2d(hid_chn),
            nn.GELU(),

            nn.Flatten(2),

            nn.Conv1d(hid_chn, hid_chn, kernel_size=(5,), padding='same', bias=False),
            nn.BatchNorm1d(hid_chn),
            nn.GELU(),
        )


        self.wavelet_decom = LiftingScheme1D(hid_chn, levels=levels)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.seBlock = SEBlock(hid_chn * (levels + 1), reduction=4)
        self.classifier = nn.Sequential(
            nn.Linear(hid_chn * (levels + 1), 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
            # nn.Softmax(dim=-1),
        )

    def forward(self, x):
        '''

        :param x: B L C
        :return:
        '''
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv1(x)
        L_list, H_list = self.wavelet_decom(x)

        f = []
        f.append(self.gap(L_list[-1]).squeeze(-1))
        for i, d in enumerate(H_list):
            f.append(self.gap(d).squeeze(-1))

        x = torch.cat(f, dim=-1)
        x = self.seBlock(x)
        return self.classifier(x), L_list, H_list

    def loss_fn(self, y_hat, y, H_list, L_list, lambda1=0.01, lambda2=0.01):
        ce = self.criterion(y_hat, y)

        # compute mean absolute of H means
        lamH = 0.0
        for H in H_list:
            lamH = lamH + H.abs().mean()
        lamH = lambda1 * lamH

        # compute mean difference of successive L means
        lamL = 0.0
        for i in range(len(L_list) - 1):
            lamL = lamL + torch.dist(L_list[i + 1].mean(), L_list[i].mean(), p=2)  # L2 norm

        lamL = lambda2 * lamL
        return ce + lamH + lamL

    def train_step(self, batch: torch.Tensor):
        x, y = batch
        x = x.to(self.device)
        y = y[:, 1].to(self.device)

        pred, L_list, H_list = self(x)

        loss = self.loss_fn(pred, y, H_list, L_list, lambda1=self.lambda1, lambda2=self.lambda2)
        return {
            'loss': loss,
            'pred': pred,
            'target': y
        }
