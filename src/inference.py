import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from .models import UniUIR, UniUIRConfig
from .datasets import UnpairedUIRDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="Stage II checkpoint (must contain LCDM/SFPG* weights)")
    p.add_argument("--input", type=str, required=True,
                   help="path to a folder of test images")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--save_depth", action="store_true")
    p.add_argument("--depth_backend", type=str, default="hf",
                   choices=["hf", "dummy"])
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8))


def depth_to_pil(d: torch.Tensor) -> Image.Image:
    arr = d.detach().clamp(0, 1).cpu().squeeze(0).numpy()
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8))


@torch.no_grad()
def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.save_depth:
        Path(args.out_dir, "depth").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = UniUIRConfig(depth_backend=args.depth_backend)
    model = UniUIR(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.depth_predictor.eval()

    dataset = UnpairedUIRDataset(args.input, pad_to=16)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    for batch in loader:
        x_lq = batch["x_lq"].to(device)
        name = batch["name"][0]
        H_orig, W_orig = batch["orig_hw"][0].item(), batch["orig_hw"][1].item()

        out = model(x_lq, mode="inference")
        x_hq = out["x_hq"].clamp(0, 1)[0, :, :H_orig, :W_orig]
        tensor_to_pil(x_hq).save(Path(args.out_dir) / name)

        if args.save_depth:
            d = out["depth"][0, :, :H_orig, :W_orig]
            depth_to_pil(d).save(Path(args.out_dir, "depth") / name)
        print(f"  -> {name}")

    print(f"[infer] done. results in {args.out_dir}")


if __name__ == "__main__":
    main()
