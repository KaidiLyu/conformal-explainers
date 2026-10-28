"""五种解释器统一返回 H×W 归因图，供校准模块直接使用。"""

import numpy as np
import torch
from captum.attr import GradientShap, InputXGradient, KernelShap, Saliency
from torch import nn
from torch.nn import functional as F


class Probabilities(nn.Module):
    """统一解释类别概率；五种方法使用相同的分类器输出定义。"""

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(x).softmax(1)


def grid_segments(height, width, grid):
    """构造规则网格作为 SHAP 特征；它与后续 SLIC 校准区域是两个概念。"""
    if not 2 <= grid <= min(height, width):
        raise ValueError("grid must be between 2 and image size")
    rows = np.arange(height) * grid // height
    cols = np.arange(width) * grid // width
    return rows[:, None] * grid + cols[None, :]


class Explainer:
    def __init__(self, model, method, *, samples=128, grid=7, noise=0.0, fastshap=None):
        self.model = model.eval()
        self.forward = Probabilities(model).eval()
        self.method, self.samples, self.grid, self.noise = method, samples, grid, noise
        self.fastshap = fastshap
        if method not in {
            "saliency",
            "input_x_gradient",
            "gradient_shap",
            "kernel_shap",
            "fastshap",
        }:
            raise ValueError(f"Unknown explainer: {method}")
        if method == "fastshap" and fastshap is None:
            raise ValueError(
                "FastSHAP requires a trained checkpoint; run train-fastshap first"
            )
        if samples < 2:
            raise ValueError("samples must be >= 2")

    def __call__(self, image, target=None):
        x = image.detach().unsqueeze(0).requires_grad_(True)
        if target is None:
            with torch.no_grad():
                target = int(self.model(x).argmax(1).item())
        # FastSHAP 的每个网格输出一个贡献值，用最近邻映射回像素网格。
        if self.method == "fastshap":
            raw = self.fastshap.shap_values(x.detach())[:, target : target + 1]
            raw = F.interpolate(
                torch.as_tensor(raw), size=image.shape[-2:], mode="nearest"
            )
            return raw[0, 0].numpy()
        if self.method == "saliency":
            a = Saliency(self.forward).attribute(x, target=target, abs=True)
        elif self.method == "input_x_gradient":
            a = InputXGradient(self.forward).attribute(x, target=target)
        elif self.method == "gradient_shap":
            # 这里使用固定零基线分布，匹配论文的 baseline removal 约定。
            a = GradientShap(self.forward).attribute(
                x,
                baselines=torch.zeros_like(x).repeat(2, 1, 1, 1),
                target=target,
                n_samples=self.samples,
                stdevs=self.noise,
            )
        else:
            groups = grid_segments(*image.shape[-2:], self.grid)
            feature_mask = torch.as_tensor(groups, device=x.device)[None, None]
            a = KernelShap(self.forward).attribute(
                x,
                baselines=0,
                target=target,
                feature_mask=feature_mask,
                n_samples=self.samples,
                perturbations_per_eval=16,
            )
            # Captum 会把同一组系数复制到组内每个像素和颜色通道。
            # 因此直接取一个通道，不能把重复的 RGB 系数再相加。
            return a[0, 0].detach().cpu().numpy()
        # 梯度方法的 RGB 贡献按通道求和；Saliency 已在上面取绝对值。
        return a[0].sum(0).detach().cpu().numpy()
