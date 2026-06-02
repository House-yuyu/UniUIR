import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from uniuir.datasets import PairedUIRDataset, UnpairedUIRDataset
from uniuir.models import UniUIR, UniUIRConfig
from uniuir.metrics import (
    psnr, ssim, LPIPSMetric,
    uciqe, uiqm,
    uranker_batch, uif_batch, uiqi_batch,
)


REF_METRICS = {"psnr", "ssim", "lpips"}
NON_REF_METRICS = {"uciqe", "uiqm", "uranker", "uif", "uiqi"}
ALL_METRICS = REF_METRICS | NON_REF_METRICS


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("paired", help="paired evaluation (compute also non-ref on the restorations)")
    pp.add_argument("--ckpt", type=str, required=True, help="UniUIR Stage II checkpoint")
    pp.add_argument("--data_root", type=str, required=True, help="contains input/ and gt/ subdirs")
    pp.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips", "uciqe", "uiqm"],
                    help=f"any of {sorted(ALL_METRICS)}")
    pp.add_argument("--depth_backend", type=str, default="hf", choices=["hf", "dummy"])
    pp.add_argument("--save_restored", type=str, default=None,
                    help="optional folder to dump restored images")
    pp.add_argument("--csv_out", type=str, default=None)
    pp.add_argument("--num_workers", type=int, default=2)

    pu = sub.add_parser("unpaired", help="non-reference evaluation on a folder of images")
    pu.add_argument("--restored_dir", type=str, required=True)
    pu.add_argument("--metrics", nargs="+", default=["uciqe", "uiqm"],
                    help=f"any of {sorted(NON_REF_METRICS)}")
    pu.add_argument("--csv_out", type=str, default=None)
    pu.add_argument("--num_workers", type=int, default=2)

    return p.parse_args()


# ---------------------------------------------------------------------------
def _safe_call(name: str, fn, *args, **kwargs):
    """Call `fn` and return (ok, value). If it raises, print a warning and return (False, None)."""
    try:
        return True, fn(*args, **kwargs)
    except Exception as e:
        print(f"  [warn] {name} unavailable: {e}", file=sys.stderr)
        return False, None


# ---------------------------------------------------------------------------
def _setup_lpips(metrics: List[str], device: torch.device) -> Optional[LPIPSMetric]:
    if "lpips" not in metrics:
        return None
    try:
        return LPIPSMetric(net="alex", device=device)
    except Exception as e:
        print(f"  [warn] LPIPS unavailable: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
def _per_image_metrics(restored: torch.Tensor,
                       gt: Optional[torch.Tensor],
                       metrics: List[str],
                       lpips_fn: Optional[LPIPSMetric]) -> Dict[str, float]:
    """Compute the requested metrics for a single restored image.

    `restored`, `gt`: (1, 3, H, W) in [0, 1].
    """
    result: Dict[str, float] = {}
    if gt is not None:
        if "psnr" in metrics:
            result["psnr"] = psnr(restored, gt)
        if "ssim" in metrics:
            result["ssim"] = ssim(restored, gt)
        if "lpips" in metrics and lpips_fn is not None:
            ok, v = _safe_call("lpips", lpips_fn, restored, gt)
            if ok:
                result["lpips"] = v
    if "uciqe" in metrics:
        ok, v = _safe_call("uciqe", uciqe, restored[0])
        if ok:
            result["uciqe"] = v
    if "uiqm" in metrics:
        ok, v = _safe_call("uiqm", uiqm, restored[0])
        if ok:
            result["uiqm"] = v
    if "uranker" in metrics:
        ok, v = _safe_call("uranker", uranker_batch, restored)
        if ok:
            result["uranker"] = v[0]
    if "uif" in metrics:
        ok, v = _safe_call("uif", uif_batch, restored)
        if ok:
            result["uif"] = v[0]
    if "uiqi" in metrics:
        ok, v = _safe_call("uiqi", uiqi_batch, restored)
        if ok:
            result["uiqi"] = v[0]
    return result


# ---------------------------------------------------------------------------
def _write_csv(rows: List[Dict[str, float]], path: str, header: List[str]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  -> wrote {path}")


def _print_summary(rows: List[Dict[str, float]], metrics: List[str]):
    print("\n=== per-metric averages ===")
    for m in metrics:
        vals = [r[m] for r in rows if m in r and r[m] is not None]
        if not vals:
            print(f"  {m:<8} : n/a")
            continue
        mean = sum(vals) / len(vals)
        print(f"  {m:<8} : mean={mean:.4f}   n={len(vals)}")


def _save_image(t: torch.Tensor, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255 + 0.5).astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
def evaluate_paired(args):
    metrics = [m.lower() for m in args.metrics]
    for m in metrics:
        if m not in ALL_METRICS:
            raise SystemExit(f"unknown metric: {m}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    cfg = UniUIRConfig(depth_backend=args.depth_backend)
    model = UniUIR(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.depth_predictor.eval()

    dataset = PairedUIRDataset(args.data_root, crop_size=128, augment=False)
    # augment=False keeps full-resolution images, but PairedUIRDataset only crops
    # when augment=True. Verify by reading one sample's full size.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    lpips_fn = _setup_lpips(metrics, device)

    rows = []
    for batch in loader:
        x_lq = batch["x_lq"].to(device)
        x_gt = batch["x_gt"].to(device)
        name = batch["name"][0]

        # Pad to multiple of 16 for the U-shape network.
        _, _, H, W = x_lq.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            x_lq = torch.nn.functional.pad(x_lq, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            out = model(x_lq, mode="inference")
        x_hq = out["x_hq"][:, :, :H, :W].clamp(0, 1)

        if args.save_restored:
            _save_image(x_hq[0], os.path.join(args.save_restored, name))

        m_dict = _per_image_metrics(x_hq, x_gt, metrics, lpips_fn)
        m_dict["name"] = name
        rows.append(m_dict)
        print(f"  {name}: " + " ".join(f"{k}={v:.4f}" for k, v in m_dict.items() if k != "name"))

    _print_summary(rows, metrics)
    if args.csv_out:
        _write_csv(rows, args.csv_out, header=["name"] + metrics)


# ---------------------------------------------------------------------------
def evaluate_unpaired(args):
    metrics = [m.lower() for m in args.metrics]
    for m in metrics:
        if m not in NON_REF_METRICS:
            raise SystemExit(f"non-ref metrics only; got: {m}")

    dataset = UnpairedUIRDataset(args.restored_dir, pad_to=1)  # no padding -- already restored
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for batch in loader:
        x = batch["x_lq"].to(device).clamp(0, 1)
        name = batch["name"][0]
        m_dict = _per_image_metrics(x, gt=None, metrics=metrics, lpips_fn=None)
        m_dict["name"] = name
        rows.append(m_dict)
        print(f"  {name}: " + " ".join(f"{k}={v:.4f}" for k, v in m_dict.items() if k != "name"))

    _print_summary(rows, metrics)
    if args.csv_out:
        _write_csv(rows, args.csv_out, header=["name"] + metrics)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.mode == "paired":
        evaluate_paired(args)
    elif args.mode == "unpaired":
        evaluate_unpaired(args)


if __name__ == "__main__":
    main()
