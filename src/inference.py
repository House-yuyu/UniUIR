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
    p.add_argument("--prior_mode", type=str, default="diffusion",
                   choices=["direct", "diffusion"])
    p.add_argument(
        "--diffusion_anchor_strength",
        type=float,
        default=None,
        help="override checkpoint direct-to-LCDM anchor strength",
    )
    p.add_argument(
        "--diffusion_stochastic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override checkpoint reverse-step stochasticity",
    )
    p.add_argument(
        "--diffusion_state_clip",
        type=float,
        default=None,
        help="override checkpoint normalized LCDM state clipping",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="per-image Gaussian seed; matches Stage-II validation by default",
    )
    p.add_argument(
        "--diffusion_samples",
        type=int,
        default=None,
        help="number of Gaussian LCDM priors averaged before restoration",
    )
    p.add_argument(
        "--tta",
        action="store_true",
        help="average original/horizontal/vertical/both flip predictions",
    )
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

    ckpt = torch.load(args.ckpt, map_location="cpu")
    checkpoint_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    anchor_strength = args.diffusion_anchor_strength
    if anchor_strength is None:
        anchor_strength = checkpoint_args.get(
            "diffusion_anchor_strength",
            1.0,
        )
    diffusion_stochastic = args.diffusion_stochastic
    if diffusion_stochastic is None:
        diffusion_stochastic = checkpoint_args.get(
            "diffusion_stochastic",
            True,
        )
    diffusion_state_clip = args.diffusion_state_clip
    if diffusion_state_clip is None:
        diffusion_state_clip = checkpoint_args.get(
            "diffusion_state_clip",
            0.0,
        )
    diffusion_samples = args.diffusion_samples
    if diffusion_samples is None:
        diffusion_samples = checkpoint_args.get("diffusion_eval_samples", 1)
    if diffusion_samples < 1:
        raise ValueError("--diffusion_samples must be at least 1")
    cfg = UniUIRConfig(
        depth_backend=args.depth_backend,
        diffusion_state_clip=diffusion_state_clip,
    )
    model = UniUIR(cfg).to(device)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    required_prefixes = ("backbone.", "sfpg_star.", "lcdm.")
    missing_prefixes = [
        prefix for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in sd)
    ]
    if missing_prefixes:
        raise RuntimeError(
            "checkpoint does not look like a Stage II checkpoint; "
            f"missing prefixes: {missing_prefixes}"
        )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    required_missing = [
        key for key in missing
        if key.startswith(required_prefixes)
    ]
    if required_missing or unexpected:
        raise RuntimeError(
            "failed to load Stage II checkpoint cleanly: "
            f"missing={required_missing}, unexpected={unexpected}"
        )
    model.eval()
    model.depth_predictor.eval()

    dataset = UnpairedUIRDataset(args.input, pad_to=16)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(
        "[infer] "
        f"prior_mode={args.prior_mode} "
        f"anchor_strength={anchor_strength} "
        f"stochastic={diffusion_stochastic} "
        f"state_clip={diffusion_state_clip} samples={diffusion_samples} "
        f"seed={args.seed}"
    )

    for image_index, batch in enumerate(loader):
        x_lq = batch["x_lq"].to(device)
        name = batch["name"][0]
        H_orig, W_orig = batch["orig_hw"][0].item(), batch["orig_hw"][1].item()

        if args.tta:
            predictions = []
            out = None
            for variant_index, dims in enumerate(
                [(), (-1,), (-2,), (-2, -1)]
            ):
                transformed = torch.flip(x_lq, dims=dims) if dims else x_lq
                generator = None
                if args.prior_mode == "diffusion":
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        args.seed + image_index * 4 + variant_index
                    )
                variant = model(
                    transformed,
                    mode="inference",
                    prior_mode=args.prior_mode,
                    sample_generator=generator,
                    diffusion_anchor_strength=anchor_strength,
                    diffusion_stochastic=diffusion_stochastic,
                    diffusion_samples=diffusion_samples,
                )
                prediction = variant["x_hq"]
                if dims:
                    prediction = torch.flip(prediction, dims=dims)
                predictions.append(prediction)
                if not dims:
                    out = variant
            x_hq_batch = torch.stack(predictions, dim=0).mean(dim=0)
        else:
            generator = None
            if args.prior_mode == "diffusion":
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + image_index)
            out = model(
                x_lq,
                mode="inference",
                prior_mode=args.prior_mode,
                sample_generator=generator,
                diffusion_anchor_strength=anchor_strength,
                diffusion_stochastic=diffusion_stochastic,
                diffusion_samples=diffusion_samples,
            )
            x_hq_batch = out["x_hq"]
        x_hq = x_hq_batch.clamp(0, 1)[0, :, :H_orig, :W_orig]
        tensor_to_pil(x_hq).save(Path(args.out_dir) / name)

        if args.save_depth:
            d = out["depth"][0, :, :H_orig, :W_orig]
            depth_to_pil(d).save(Path(args.out_dir, "depth") / name)
        print(f"  -> {name}")

    print(f"[infer] done. results in {args.out_dir}")


if __name__ == "__main__":
    main()
