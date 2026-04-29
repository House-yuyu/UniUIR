import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str) -> List[str]:
    folder = Path(folder)
    files = [str(p) for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)


# ---------------------------------------------------------------------------
class PairedUIRDataset(Dataset):
    def __init__(self, root: str, crop_size: int = 128, augment: bool = True):
        self.input_dir = Path(root) / "input"
        self.gt_dir = Path(root) / "GT"
        assert self.input_dir.exists() and self.gt_dir.exists(), \
            f"expected `input/` and `GT/` under {root}"
        self.names = []
        for p in list_images(self.input_dir):
            name = Path(p).name
            if (self.gt_dir / name).exists():
                self.names.append(name)
        self.crop_size = crop_size
        self.augment = augment
        if not self.names:
            raise RuntimeError(f"no paired files found in {root}")

    def __len__(self):
        return len(self.names)

    def _random_crop(self, lq: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, H, W = lq.shape
        s = self.crop_size
        if H < s or W < s:
            # Pad if image smaller than crop window.
            ph = max(0, s - H)
            pw = max(0, s - W)
            lq = torch.nn.functional.pad(lq, (0, pw, 0, ph), mode="reflect")
            gt = torch.nn.functional.pad(gt, (0, pw, 0, ph), mode="reflect")
            _, H, W = lq.shape
        i = random.randint(0, H - s)
        j = random.randint(0, W - s)
        return lq[:, i:i + s, j:j + s], gt[:, i:i + s, j:j + s]

    def __getitem__(self, idx):
        name = self.names[idx]
        lq = pil_to_tensor(Image.open(self.input_dir / name))
        gt = pil_to_tensor(Image.open(self.gt_dir / name))
        if self.augment:
            lq, gt = self._random_crop(lq, gt)
            if random.random() < 0.5:
                lq = torch.flip(lq, dims=[-1])
                gt = torch.flip(gt, dims=[-1])
            if random.random() < 0.5:
                lq = torch.flip(lq, dims=[-2])
                gt = torch.flip(gt, dims=[-2])
        return {"x_lq": lq, "x_gt": gt, "name": name}


# ---------------------------------------------------------------------------
class UnpairedUIRDataset(Dataset):
    """For non-reference test sets. Pads to multiple-of-`pad_to` for the network."""
    def __init__(self, root: str, pad_to: int = 16):
        self.files = list_images(root)
        self.pad_to = pad_to
        if not self.files:
            raise RuntimeError(f"no images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = pil_to_tensor(Image.open(path))
        _, H, W = img.shape
        ph = (self.pad_to - H % self.pad_to) % self.pad_to
        pw = (self.pad_to - W % self.pad_to) % self.pad_to
        if ph or pw:
            img = torch.nn.functional.pad(img, (0, pw, 0, ph), mode="reflect")
        return {"x_lq": img, "name": Path(path).name, "orig_hw": (H, W)}
