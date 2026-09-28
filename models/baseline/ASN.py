"""
"Attentive Siamese Networks for Automatic Modulation Classification Based on Multitiming Constellation Diagrams"
https://ieeexplore.ieee.org/abstract/document/9667215

reproduced by asme

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .. import BaseModel

# === 1. 特征提取模块（简化 VGG16） ===
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            # nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            # nn.AdaptiveAvgPool2d((16, 16)),  # 保持输出大小为 16x16
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)  # 输出特征维度
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)


# === 2. 注意力模块（Attention Inference） ===
class AttentionModule(nn.Module):
    def __init__(self, num_branches=8, feature_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(num_branches * feature_dim, 32)
        self.fc2 = nn.Linear(32, num_branches)

    def forward(self, features):  # [B, K, D]
        B, K, D = features.shape
        fused = features.view(B, -1)  # [B, K*D]
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(fused))))  # [B, K]
        weights = weights.unsqueeze(-1)  # [B, K, 1]
        return features * weights  # 加权融合


# === 3. 分类模块 ===
class Classifier(nn.Module):
    def __init__(self, feature_dim=10, num_branches=8, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, weighted_features):  # [B, K, D]
        summed = weighted_features.sum(dim=1)  # [B, D]

        return self.fc(summed)


# === 4. ASN 总体结构 ===
class AttentiveSiameseNetwork(BaseModel):
    def __init__(self, in_chn=1, num_branches=8, feature_dim=10, num_classes=10):
        super().__init__()
        self.num_branches = num_branches

        self.feature_extractor = FeatureExtractor(in_chn, num_classes) # feature_dim替换为num_classes 对应原文
        self.attention = AttentionModule(num_branches, num_classes)
        self.classifier = Classifier(num_classes, num_branches, num_classes)

    def forward(self, x):  # x: [B, K, C, H, W]
        B, K, C, H, W = x.shape
        x = rearrange(x, 'b k c h w -> (b k) c h w')  # [B*K, C, H, W]
        features = self.feature_extractor(x)  # 每个offset的特征
        features = rearrange(features, '(b k) d-> b k d', b=B)

        features = self.attention(features)  # [B, K, D]
        logits = self.classifier(features)  # [B, num_classes]
        return logits



if __name__ == "__main__":
    # 模拟 batch 输入：8 个不同 offset 的 32x32 星座图
    x = torch.randn(16, 8, 1, 32, 32)  # [batch=16, K=8, channel=3, height=32, width=32]
    model = AttentiveSiameseNetwork(in_chn=1)
    logits = model(x)  # [16, num_classes=10]
    print(logits.shape)  # torch.Size([16, 10])
