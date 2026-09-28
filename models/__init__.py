#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2024-04-11 10:09

@author: AS ME
"""
from .BaseModel import BaseModel
from .baseline import *

from .WaveletCNN import WaveletCNN, FixedWaveletCNN

Model_Dict = {
    "AvgNet": AvgNet,
    "ConvMixer": ConvMixer,
    "FEA_T": FEA_T,
    "LSTM": LSTM,
    "MCLDNN": MCLDNN,
    "SigNet": SigNet,
    "CTDNN": CTDNN,
    "SigFormer": SigFormer,
    "MCFormer": MCformer,
    "WAN": WAN,
    "ASN": ASN,
    "ResNet": ResNet,
    "STF_GCN": STF_GCN,
    "IQFormer": IQFormer,
    "AWN": AWN,
    "IQGCA": IQGCA,
    "ULCNN": ULCNN,
    "Conv2Former": Conv2Former,
    "AWN_official": AWN_official,
    "ArF_AMC": ArF_AMC,
    "DNNTF": DNNTF,
    "CNN1D": CNN1D,
    'WaveletCNN': WaveletCNN,
    'FixedWaveletCNN': FixedWaveletCNN,
    'ALWNN':ALWNN,
    'AMC_Net': AMC_Net
}
