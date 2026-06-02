"""
Wrappers for the three learned / complex underwater quality metrics:

  - URanker [Guo et al., AAAI 2023]  -- official: https://github.com/RQ-Wu/UnderwaterRanker
  - UIF     [Zheng et al., TIP 2022] -- official MATLAB code is provided by the authors;
                                        a Python port is at https://github.com/z21110008/UIF
  - UIQI    [Liu et al., TMM 2024]   -- the official code is hosted on the project page.

These three metrics are non-trivial to reimplement faithfully: URanker uses a
pretrained network whose weights must match the paper exactly; UIF and UIQI both
rely on long pipelines of natural-scene-statistics features whose published
formulae have several conventions that differ between code and paper.

Rather than risk subtle numerical mismatches, this module provides a thin
interface layer: each metric is a callable that

  (a) loads the *official* model / code on first use, and
  (b) accepts a tensor in (B, 3, H, W) in [0, 1] and returns a list of floats.

Use the `register_*` functions at the bottom to plug in your own implementation
if you have a custom version, or set the `*_PATH` environment variables before
import to point at the official checkpoints / source trees.
"""

import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# URanker
# ---------------------------------------------------------------------------
class _URankerLoader:
    """Lazy loader for the official URanker model.

    Steps to enable:
      1. clone https://github.com/RQ-Wu/UnderwaterRanker  -- e.g. into ./third_party/UnderwaterRanker
      2. download the released checkpoint (URanker.pth) to that folder
      3. either
           - put the repo on PYTHONPATH and set URANKER_CKPT=<path/to/URanker.pth>, OR
           - call `register_uranker(my_callable)` with your own forward function.
    """
    def __init__(self):
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None

    def _build(self, device: torch.device):
        ckpt_path = os.environ.get("URANKER_CKPT", None)
        if ckpt_path is None:
            raise RuntimeError(
                "URanker: please set environment variable URANKER_CKPT to the path "
                "of URanker.pth from https://github.com/RQ-Wu/UnderwaterRanker, "
                "and ensure the repo is on PYTHONPATH so that "
                "`from net.uranker import NIMA` is importable.")
        # Defer import to runtime; the module name `net.uranker` matches the
        # official repo layout. If your local copy uses a different name, edit
        # this import or use register_uranker(...).
        try:
            from net.uranker import NIMA  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "URanker: could not import `net.uranker.NIMA`. Make sure the "
                "official repo is on PYTHONPATH.") from e
        net = NIMA().to(device)
        sd = torch.load(ckpt_path, map_location=device)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict(sd, strict=False)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self._model = net
        self._device = device

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> List[float]:
        device = batch.device
        if self._model is None or self._device != device:
            self._build(device)
        # URanker expects ImageNet-normalised input.
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x = (batch.clamp(0, 1) - mean) / std
        score = self._model(x)        # (B, 1) or (B,)
        return score.flatten().detach().cpu().tolist()


_uranker_callable: Optional[Callable] = None


def register_uranker(fn: Callable):
    """Override the default URanker loader (e.g. with a custom implementation)."""
    global _uranker_callable
    _uranker_callable = fn


def uranker_batch(batch: torch.Tensor) -> List[float]:
    """Score a batch of images with URanker. Returns a list of length B."""
    global _uranker_callable
    if _uranker_callable is None:
        _uranker_callable = _URankerLoader()
    return _uranker_callable(batch)


# ---------------------------------------------------------------------------
# UIF
# ---------------------------------------------------------------------------
class _UIFLoader:
    """Lazy loader for UIF (Zheng et al., TIP 2022).

    The official authors release a MATLAB implementation; a faithful Python port
    is maintained at https://github.com/z21110008/UIF (also referenced by the
    UIQI paper). To enable:
      - clone the port and put it on PYTHONPATH, or
      - set UIF_MODULE=<dotted.path.to.module> whose top-level provides
        `score(rgb_uint8: np.ndarray) -> float`.
    """
    def __init__(self):
        self._fn: Optional[Callable] = None

    def _build(self):
        mod_name = os.environ.get("UIF_MODULE", None)
        if mod_name is None:
            raise RuntimeError(
                "UIF: please install the official Python port "
                "(https://github.com/z21110008/UIF) and set environment "
                "variable UIF_MODULE to the importable module name that "
                "exposes `score(rgb_uint8) -> float`.")
        import importlib
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "score"):
            raise RuntimeError(f"UIF: module {mod_name} has no `score` function.")
        self._fn = mod.score

    def __call__(self, batch: torch.Tensor) -> List[float]:
        if self._fn is None:
            self._build()
        out = []
        for i in range(batch.size(0)):
            arr = (batch[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
            out.append(float(self._fn(arr)))
        return out


_uif_callable: Optional[Callable] = None


def register_uif(fn: Callable):
    global _uif_callable
    _uif_callable = fn


def uif_batch(batch: torch.Tensor) -> List[float]:
    global _uif_callable
    if _uif_callable is None:
        _uif_callable = _UIFLoader()
    return _uif_callable(batch)


# ---------------------------------------------------------------------------
# UIQI
# ---------------------------------------------------------------------------
class _UIQILoader:
    """Lazy loader for UIQI (Liu et al., TMM 2024).

    To enable, install the official UIQI Python implementation and set
    UIQI_MODULE=<dotted.path> whose top-level exposes `score(rgb_uint8) -> float`.
    """
    def __init__(self):
        self._fn: Optional[Callable] = None

    def _build(self):
        mod_name = os.environ.get("UIQI_MODULE", None)
        if mod_name is None:
            raise RuntimeError(
                "UIQI: please install the official UIQI implementation and set "
                "environment variable UIQI_MODULE to the importable module that "
                "exposes `score(rgb_uint8) -> float`.")
        import importlib
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "score"):
            raise RuntimeError(f"UIQI: module {mod_name} has no `score` function.")
        self._fn = mod.score

    def __call__(self, batch: torch.Tensor) -> List[float]:
        if self._fn is None:
            self._build()
        out = []
        for i in range(batch.size(0)):
            arr = (batch[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
            out.append(float(self._fn(arr)))
        return out


_uiqi_callable: Optional[Callable] = None


def register_uiqi(fn: Callable):
    global _uiqi_callable
    _uiqi_callable = fn


def uiqi_batch(batch: torch.Tensor) -> List[float]:
    global _uiqi_callable
    if _uiqi_callable is None:
        _uiqi_callable = _UIQILoader()
    return _uiqi_callable(batch)
