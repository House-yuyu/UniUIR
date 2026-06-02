"""
Non-reference underwater image quality metrics: UCIQE and UIQM.

References
----------
[1] M. Yang and A. Sowmya. "An Underwater Color Image Quality Evaluation Metric."
    IEEE TIP, 2015.   -- UCIQE
[2] K. Panetta, C. Gao, and S. Agaian. "Human-Visual-System-Inspired Underwater
    Image Quality Measures." IEEE J. Oceanic Eng., 2016.   -- UIQM

These functions accept either a numpy uint8 image (H, W, 3) RGB or a torch tensor
(B, 3, H, W) in [0, 1], and return a Python float (or list of floats per item).

"""

from typing import Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_uint8_rgb(img: ArrayLike) -> np.ndarray:
    """Coerce input to a uint8 RGB ndarray of shape (H, W, 3)."""
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu()
        if x.dim() == 4:
            assert x.size(0) == 1, "use the batch helpers for multi-image input"
            x = x[0]
        # (3, H, W) -> (H, W, 3)
        x = x.permute(1, 2, 0).numpy()
        x = np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return x
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8) if img.max() <= 1.5 \
              else np.clip(img + 0.5, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def _rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB (uint8, 0..255) -> CIELab using the standard D65 conversion."""
    rgb = rgb_u8.astype(np.float64) / 255.0
    # sRGB inverse companding
    mask = rgb > 0.04045
    rgb_lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # sRGB -> XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = rgb_lin @ M.T
    # Reference white D65
    X, Y, Z = xyz[..., 0] / 0.95047, xyz[..., 1] / 1.0, xyz[..., 2] / 1.08883
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = lambda t: np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)
    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


# ---------------------------------------------------------------------------
# UCIQE
# ---------------------------------------------------------------------------
def uciqe(image: ArrayLike,
          c1: float = 0.4680, c2: float = 0.2745, c3: float = 0.2576) -> float:
    """UCIQE = c1 * sigma_chroma + c2 * contrast_L + c3 * mean_saturation.

    Implementation matches the convention used by FUnIE-GAN and most UIR papers:
      - All channels are mapped into [0, 1] before statistics are computed.
      - sigma_chroma : std of chroma C = sqrt(a^2 + b^2) / 255  (in [0, ~0.71])
      - contrast_L   : top 1% mean - bottom 1% mean of L / 100  (L is in [0, 100])
      - mean_sat     : mean of "angle-style" saturation s = C / sqrt(C^2 + L^2)
    Coefficients (0.4680, 0.2745, 0.2576) are the originals from Yang & Sowmya, 2015.
    """
    rgb = _to_uint8_rgb(image)
    lab = _rgb_to_lab(rgb)
    L = lab[..., 0]                      # in [0, 100]
    a = lab[..., 1]
    b = lab[..., 2]

    L_norm = L / 100.0
    chroma_norm = np.sqrt(a ** 2 + b ** 2) / 255.0

    sigma_chroma = float(chroma_norm.std())

    L_flat = np.sort(L_norm.flatten())
    k = max(1, int(0.01 * L_flat.size))
    contrast_L = float(L_flat[-k:].mean() - L_flat[:k].mean())

    sat = chroma_norm / np.sqrt(chroma_norm ** 2 + L_norm ** 2 + 1e-12)
    mean_sat = float(sat.mean())

    return c1 * sigma_chroma + c2 * contrast_L + c3 * mean_sat


# ---------------------------------------------------------------------------
# UIQM components
# ---------------------------------------------------------------------------
def _uicm(rgb_u8: np.ndarray, alpha_l: float = 0.1, alpha_r: float = 0.1) -> float:
    """Underwater Image Colorfulness Measure.

    Asymmetric alpha-trimmed statistics on RG = R-G and YB = (R+G)/2 - B.
    """
    rgb = rgb_u8.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    RG = (R - G).flatten()
    YB = ((R + G) / 2.0 - B).flatten()

    def trimmed_mean_var(x: np.ndarray) -> tuple:
        x = np.sort(x)
        N = x.size
        T_l = int(np.floor(alpha_l * N))
        T_r = int(np.ceil((1.0 - alpha_r) * N))
        T_r = max(T_l + 1, min(T_r, N))
        x_t = x[T_l:T_r]
        return float(x_t.mean()), float(x_t.var())

    mu_RG, var_RG = trimmed_mean_var(RG)
    mu_YB, var_YB = trimmed_mean_var(YB)
    return -0.0268 * np.sqrt(mu_RG ** 2 + mu_YB ** 2) \
           + 0.1586 * np.sqrt(var_RG + var_YB)


def _eme(channel: np.ndarray, block: int = 8) -> float:
    """Standard EME (Measure of Enhancement) on a 2-D channel."""
    H, W = channel.shape
    nH, nW = H // block, W // block
    if nH == 0 or nW == 0:
        return 0.0
    eme_sum = 0.0
    for i in range(nH):
        for j in range(nW):
            blk = channel[i * block:(i + 1) * block, j * block:(j + 1) * block]
            mx, mn = float(blk.max()), float(blk.min())
            if mn < 1e-6 or mx <= mn:
                continue
            eme_sum += np.log(mx / mn)
    return 2.0 / (nH * nW) * eme_sum


def _sobel_edge(channel: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude with edge-preserving padding."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    pad = np.pad(channel, 1, mode="edge")
    Gx = (pad[:-2, :-2] * Kx[0, 0] + pad[:-2, 1:-1] * Kx[0, 1] + pad[:-2, 2:] * Kx[0, 2] +
          pad[1:-1, :-2] * Kx[1, 0] + pad[1:-1, 1:-1] * Kx[1, 1] + pad[1:-1, 2:] * Kx[1, 2] +
          pad[2:, :-2] * Kx[2, 0] + pad[2:, 1:-1] * Kx[2, 1] + pad[2:, 2:] * Kx[2, 2])
    Gy = (pad[:-2, :-2] * Ky[0, 0] + pad[:-2, 1:-1] * Ky[0, 1] + pad[:-2, 2:] * Ky[0, 2] +
          pad[1:-1, :-2] * Ky[1, 0] + pad[1:-1, 1:-1] * Ky[1, 1] + pad[1:-1, 2:] * Ky[1, 2] +
          pad[2:, :-2] * Ky[2, 0] + pad[2:, 1:-1] * Ky[2, 1] + pad[2:, 2:] * Ky[2, 2])
    return np.sqrt(Gx ** 2 + Gy ** 2)


def _uism(rgb_u8: np.ndarray, block: int = 8) -> float:
    """Underwater Image Sharpness Measure: weighted EME of edge-multiplied channels."""
    rgb = rgb_u8.astype(np.float64)
    weights = (0.299, 0.587, 0.114)  # standard luma weights
    score = 0.0
    for c in range(3):
        ch = rgb[..., c]
        edge = _sobel_edge(ch)
        # Multiply edges by the original channel (paper's convention).
        edge_img = (edge * ch).astype(np.float64)
        score += weights[c] * _eme(edge_img, block=block)
    return score


def _logamee(gray: np.ndarray, block: int = 8) -> float:
    """logAMEE used in UIConM: PLIP-style log-average of (Imax-Imin)/(Imax+Imin)."""
    H, W = gray.shape
    nH, nW = H // block, W // block
    if nH == 0 or nW == 0:
        return 0.0
    s = 0.0
    cnt = 0
    for i in range(nH):
        for j in range(nW):
            blk = gray[i * block:(i + 1) * block, j * block:(j + 1) * block]
            mx, mn = float(blk.max()), float(blk.min())
            num = mx - mn
            den = mx + mn
            if den < 1e-6 or num < 1e-6:
                continue
            r = num / den
            s += r * np.log(r + 1e-12)
            cnt += 1
    if cnt == 0:
        return 0.0
    return -1.0 / cnt * s


def _uiconm(rgb_u8: np.ndarray, block: int = 8) -> float:
    """Underwater Image Contrast Measure via logAMEE on grayscale."""
    gray = rgb_u8.astype(np.float64).mean(axis=-1) / 255.0
    return _logamee(gray, block=block)


def uiqm(image: ArrayLike,
         c1: float = 0.0282, c2: float = 0.2953, c3: float = 3.5753,
         block: int = 8) -> float:
    """UIQM = c1 * UICM + c2 * UISM + c3 * UIConM (Panetta et al., 2016)."""
    rgb = _to_uint8_rgb(image)
    return c1 * _uicm(rgb) + c2 * _uism(rgb, block=block) + c3 * _uiconm(rgb, block=block)


def uciqe_batch(batch: torch.Tensor) -> list:
    """batch: (B, 3, H, W) in [0, 1]."""
    return [uciqe(batch[i]) for i in range(batch.size(0))]


def uiqm_batch(batch: torch.Tensor) -> list:
    return [uiqm(batch[i]) for i in range(batch.size(0))]
