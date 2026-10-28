"""论文第 3 节的阈值校准：把归因图变成需要保留的图像区域。

注意：保留更多像素不一定让分类结果保持不变，因此不能使用二分搜索。
论文分位数规则用于复现，不能直接当成对任意分类器成立的 fidelity 保证。
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MaskConfig:
    """variant 指定校准方式；rho_mode 明确 rho 是比例还是像素数量。"""

    variant: str = "pixelwise"
    rho: float = 0.5
    rho_mode: str = "fraction"

    def __post_init__(self):
        if self.variant not in {"pixelwise", "superpixels", "scaled"}:
            raise ValueError("Unknown calibration variant")
        if self.rho_mode not in {"fraction", "count"}:
            raise ValueError("rho_mode must be fraction or count")
        if not np.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be positive and finite")
        if self.rho_mode == "fraction" and self.rho > 1:
            raise ValueError("Fraction rho must be in (0, 1]")
        if self.rho_mode == "count" and self.rho != int(self.rho):
            raise ValueError("Count rho must be an integer")


def prepare_scores(scores, config):
    """将归因图转为二维数组；Scaled Values 按每张图单独标准化。"""
    a = np.asarray(scores, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected a finite H x W attribution map")
    if config.variant == "scaled":
        # 常数归因图没有方差，统一置零，避免除零产生 NaN。
        sd = a.std()
        a = (a - a.mean()) / sd if sd > 1e-12 else np.zeros_like(a)
    return a


def make_mask(scores, threshold, config, segments=None):
    """返回 H×W 布尔掩码，True 表示保留该位置的全部颜色通道。"""
    a = prepare_scores(scores, config)
    if np.isnan(threshold) or threshold == np.inf:
        raise ValueError("Invalid threshold")
    # -inf 是明确的全图回退标志；像素少于 count rho 的小区域也必须保留。
    if threshold == -np.inf:
        return np.ones(a.shape, dtype=bool)
    selected = a >= threshold
    if config.variant == "pixelwise":
        return selected
    if segments is None or np.shape(segments) != a.shape:
        raise ValueError("Superpixel variants require a matching segmentation")
    # 将任意超像素编号压缩成连续编号，再统计各区域中超过阈值的像素。
    _, ids = np.unique(segments, return_inverse=True)
    ids = ids.reshape(a.shape)
    counts = np.bincount(ids.ravel())
    overlap = np.bincount(ids.ravel(), weights=selected.ravel())
    criterion = overlap / counts if config.rho_mode == "fraction" else overlap
    return (criterion >= config.rho)[ids]


def candidate_thresholds(scores, config, steps=100):
    """在归因值分位数上搜索，去重以避免重复推理；包含最小值和最大值。"""
    if steps < 2:
        raise ValueError("At least two threshold steps are required")
    a = prepare_scores(scores, config)
    return np.unique(np.quantile(a, np.linspace(0, 1, steps + 1)))


@torch.no_grad()
def conformity_score(
    model, image, scores, config, segments=None, steps=100, batch_size=16
):
    """寻找仍保持原图预测的最大候选阈值，对应公式 (3)/(4)。

    遍历全部候选：分类结果可能随区域增加先改变、再恢复。
    分批推理控制内存；比较对象是原图预测，不是真实标签。
    """
    if image.ndim != 3 or batch_size < 1:
        raise ValueError("Expected C x H x W image and positive batch size")
    model.eval()
    target = model(image[None]).argmax(1).item()
    thresholds = candidate_thresholds(scores, config, steps)
    best = -np.inf
    for start in range(0, len(thresholds), batch_size):
        ts = thresholds[start : start + batch_size]
        masks = np.stack([make_mask(scores, t, config, segments) for t in ts])
        mask_tensor = torch.as_tensor(masks, device=image.device, dtype=image.dtype)
        predictions = model(image[None] * mask_tensor[:, None]).argmax(1)
        ok = predictions.cpu().numpy() == target
        if ok.any():
            best = max(best, float(ts[ok].max()))
    return best


def calibrate_threshold(scores, confidence, rule="paper"):
    """用校准集分数计算全局阈值，测试图片不参与此步骤。

    paper：直接实现正文第 3.1 节的“计数加一”不等式。
    conservative：采用 floor((n+1)*alpha) 的下侧次序统计量；样本不足
    则保留整图。该选项只调整分位数秩，不解决分类结果的非单调性。
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not len(s) or np.isnan(s).any() or np.isposinf(s).any():
        raise ValueError("Nonempty conformity scores (finite or -inf) required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if rule == "paper":
        candidates, counts = np.unique(s, return_counts=True)
        tails = np.cumsum(counts[::-1])[::-1]
        valid = (tails + 1) / (len(s) + 1) >= confidence - 1e-12
        return float(candidates[valid][-1])
    if rule == "conservative":
        rank = int(np.floor((len(s) + 1) * (1 - confidence) + 1e-12))
        return float(np.sort(s)[rank - 1]) if rank >= 1 else -np.inf
    raise ValueError("Unknown quantile rule")


def wilson_interval(successes, n, z=1.959963984540054):
    """计算预测一致率的 95% Wilson 区间，避免小样本结果看起来过于确定。"""
    if n < 1:
        raise ValueError("At least one observation required")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)
