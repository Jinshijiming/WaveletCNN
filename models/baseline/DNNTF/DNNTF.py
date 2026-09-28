from ... import BaseModel

import torch.nn as nn
import torch.nn.functional as F
from .cal_feature_torch import calculate_traditional_features_batch


class DNNTF(BaseModel):
    def __init__(self, input_size=228, dropout=0.3, num_classes=12):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 96)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(96, 64)
        self.classfier = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x:IQ data
        """
        x = calculate_traditional_features_batch(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.classfier(x)
        return x
