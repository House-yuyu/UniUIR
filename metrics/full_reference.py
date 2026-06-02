"""
Full-reference metrics for paired evaluation: PSNR, SSIM, LPIPS.

Inputs throughout this file are torch.Tensor in [0, 1] of shape (B, 3, H, W).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Mean PSNR over the batch. Inputs in [0, max_val]."""
    pred = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    return (10 * torch.log10(max_val ** 2 / (mse + 1e-12))).mean().item()


# ---------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    win2d = g.t() @ g
    return win2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
         window_size: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM over the batch (per-channel mean, common convention)."""
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
class LPIPSMetric(nn.Module):
    """Wrapper around the official `lpips` package (Zhang et al. CVPR 2018).

    `pip install lpips`. The default network is AlexNet, matching the LPIPS
    numbers most papers report.
    """
    def __init__(self, net: str = "alex", device: str = "cuda"):
        super().__init__()
        try:
            import lpips
        except ImportError as e:
            raise ImportError("LPIPS requires `pip install lpips`.") from e
        self.lpips = lpips.LPIPS(net=net, verbose=False).to(device)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        # LPIPS expects inputs in [-1, 1].
        pred = pred.clamp(0, 1) * 2 - 1
        target = target.clamp(0, 1) * 2 - 1
        d = self.lpips(pred, target)
        return d.mean().item()
