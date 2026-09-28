import numpy as np

from dataset.dataset import RML_2016a, RML_2016b, RML_2018a, Sig2019_12,TechRec
import torch
from os.path import join, dirname
import yaml
from scipy.signal import stft

with open(join(dirname(__file__), 'conf.yaml'), 'r') as file:
    cfg = yaml.safe_load(file)


class RML_2016a_stft(RML_2016a):
    def __init__(self, flag='train', use_debug=0, snr='all', norm=None, transform=None, use_transform=True):
        super().__init__(flag=flag, snr=snr, use_debug=use_debug, norm=norm, transform=transform,
                         use_transform=use_transform)

    def __getitem__(self, idx):
        item = self.ds[idx]
        data = item['data']
        label = item['label']
        _, _, stp = stft(data[0], 1.0, 'blackman', 31, 30, 128)
        stp = np.abs(stp)
        data = self.T(data)

        data = torch.cat([data, torch.tensor(stp[:32])], dim=0)
        return data, torch.tensor(label)


class RML_2016b_stft(RML_2016b):
    def __init__(self, flag='train', use_debug=0, snr='all', norm=None, transform=None, use_transform=True):
        super().__init__(flag=flag, snr=snr, use_debug=use_debug, norm=norm, transform=transform,
                         use_transform=use_transform)

    def __getitem__(self, idx):
        item = self.ds[idx]
        data = item['data']
        label = item['label']
        _, _, stp = stft(data[0], 1.0, 'blackman', 31, 30, 128)
        stp = np.abs(stp)
        data = self.T(data)

        data = torch.cat([data, torch.tensor(stp[:32])], dim=0)
        return data, torch.tensor(label)


class RML_2018a_stft(RML_2018a):
    def __init__(self, flag='train', use_debug=0, snr='all', norm=None, transform=None, use_transform=True):
        super().__init__(flag=flag, snr=snr, use_debug=use_debug, norm=norm, transform=transform,
                         use_transform=use_transform)

    def __getitem__(self, idx):
        item = self.ds[idx]
        data = item['data']
        label = item['label']
        _, _, stp = stft(data[0], 1.0, 'blackman', 31, 30, 128)
        stp = np.abs(stp)
        data = self.T(data)

        data = torch.cat([data, torch.tensor(stp[:32])], dim=0)
        return data, torch.tensor(label)

class TechRec_stft(TechRec):
    def __init__(self, flag='train', use_debug=0, snr='all', norm=None, transform=None, use_transform=True):
        super().__init__(flag=flag, snr=snr, use_debug=use_debug, norm=norm, transform=transform,
                         use_transform=use_transform)

    def __getitem__(self, idx):
        item = self.ds[idx]
        data = item['data']
        label = item['label']
        _, _, stp = stft(data[0], 1.0, 'blackman', 31, 30, 128)
        stp = np.abs(stp)
        data = self.T(data)

        data = torch.cat([data, torch.tensor(stp[:32])], dim=0)
        return data, torch.tensor(label)


class Sig2019_stft(Sig2019_12):
    def __init__(self, flag='train', use_debug=0, snr='all', norm=None, transform=None, use_transform=True):
        super().__init__(flag=flag, snr=snr, use_debug=use_debug, norm=norm, transform=transform,
                         use_transform=use_transform)

    def __getitem__(self, idx):
        item = self.ds[idx]
        data = item['data']
        label = item['label']
        _, _, stp = stft(data[0], 1.0, 'blackman', 31, 30, 128)
        stp = np.abs(stp)
        data = self.T(data)

        data = torch.cat([data, torch.tensor(stp[:32])], dim=0)
        return data, torch.tensor(label)


if __name__ == '__main__':
    dataset = RML_2018a_stft(flag='train', snr='all', use_debug=0, norm=True)
    print(len(dataset))
    data, label = dataset[0]
    print(data.shape, label)
