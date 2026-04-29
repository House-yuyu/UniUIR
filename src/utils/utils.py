import os
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, v, n=1):
        self.sum += float(v) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# ---------------------------------------------------------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Per-batch mean PSNR. Inputs in [0, max_val]."""
    pred = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    p = 10 * torch.log10(max_val ** 2 / (mse + 1e-12))
    return p.mean().item()


def _gaussian_window(window_size: int, sigma: float, channels: int, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
         window_size: int = 11, sigma: float = 1.5) -> float:
    pred = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    channels = pred.size(1)
    win = _gaussian_window(window_size, sigma, channels, pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, win, padding=pad, groups=channels)
    mu2 = F.conv2d(target, win, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, win, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, win, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, win, padding=pad, groups=channels) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean().item()


# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------
def cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    if step >= total_steps:
        return min_lr
    progress = step / max(1, total_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
def count_trainable_params(model) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
