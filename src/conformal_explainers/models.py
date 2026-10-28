import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_classifier(classes=10, pretrained=False):
    """替换 ResNet50 分类头；pretrained=True 时加载 ImageNet 预训练权重。"""
    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


class GridExplainer(nn.Module):
    """基础 FastSHAP 卷积网络；论文未公布完整架构，本实现属于明确记录的选择。"""

    def __init__(self, classes=10, grid=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid, grid)),
            nn.Conv2d(64, classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class SmokeClassifier(nn.Module):
    """仅用于合成数据测试的二分类器，不能用于论文结果对比。"""

    def forward(self, x):
        value = x[:, 0, :4, :4].mean((1, 2))
        return torch.stack([-value, value], dim=1)


def load_classifier(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("synthetic"):
        model = SmokeClassifier()
    else:
        model = build_classifier(len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
