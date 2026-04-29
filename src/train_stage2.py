import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .models import UniUIR, UniUIRConfig
from .losses import StageIILoss
from .datasets import PairedUIRDataset
from .utils import (AverageMeter, psnr, ssim,
                          save_checkpoint, cosine_lr, set_lr,
                          count_trainable_params)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--val_root", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="./checkpoints/stage2")
    p.add_argument("--stage1_ckpt", type=str, required=True,
                   help="Stage I checkpoint to initialize from")
    p.add_argument("--total_iters", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lambda_diff", type=float, default=1.0)
    p.add_argument("--lambda_eps", type=float, default=0.0,
                   help="optional weight for noise-prediction loss; 0 disables")
    p.add_argument("--depth_backend", type=str, default="hf",
                   choices=["hf", "dummy"])
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--ckpt_every", type=int, default=10000)
    p.add_argument("--val_every", type=int, default=10000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cycle(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    p_meter, s_meter = AverageMeter(), AverageMeter()
    for batch in loader:
        x_lq = batch["x_lq"].to(device)
        x_gt = batch["x_gt"].to(device)
        # Use inference mode (LCDM sampling) at validation -- matches deployment.
        out = model(x_lq, mode="inference")
        x_hq = out["x_hq"].clamp(0, 1)
        p_meter.update(psnr(x_hq, x_gt), n=x_lq.size(0))
        s_meter.update(ssim(x_hq, x_gt), n=x_lq.size(0))
    model.train()
    model.depth_predictor.eval()
    model.sfpg.eval()  # SFPG remains frozen
    return p_meter.avg, s_meter.avg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    train_set = PairedUIRDataset(args.data_root, crop_size=args.crop_size, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    train_iter = cycle(train_loader)

    val_loader = None
    if args.val_root:
        val_set = PairedUIRDataset(args.val_root, crop_size=args.crop_size, augment=False)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    # ---- model + load stage1 ----
    cfg = UniUIRConfig(depth_backend=args.depth_backend)
    model = UniUIR(cfg).to(device)
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[stage2] loaded stage1: missing={len(missing)} unexpected={len(unexpected)}")
    # Expected: SFPG*, LCDM keys are in `missing` (they were not trained in stage1).

    # Freeze SFPG (its output Z is the diffusion target, must stay fixed).
    model.freeze_for_stage2()

    print(f"[stage2] trainable params (M): {count_trainable_params(model):.2f}")

    # Optimize LCDM + SFPG* + MMoE-UIR (fine-tune).
    train_params = (list(model.backbone.parameters())
                    + list(model.sfpg_star.parameters())
                    + list(model.lcdm.parameters()))
    optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    criterion = StageIILoss(lambda_diff=args.lambda_diff,
                            lambda_eps=args.lambda_eps).to(device)

    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        rckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(rckpt["model"], strict=False)
        optimizer.load_state_dict(rckpt["optimizer"])
        start_iter = rckpt["iter"]
        print(f"[stage2] resumed from {args.resume} at iter {start_iter}")

    pix_m, diff_m = AverageMeter(), AverageMeter()
    t0 = time.time()
    model.train()
    model.depth_predictor.eval()
    model.sfpg.eval()  # frozen

    for it in range(start_iter, args.total_iters):
        lr = cosine_lr(it, args.total_iters, args.lr, args.min_lr)
        set_lr(optimizer, lr)

        batch = next(train_iter)
        x_lq = batch["x_lq"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)

        out = model(x_lq, x_gt, mode="stage2")
        loss, log = criterion(out["x_hq"], x_gt, out["Z"], out["Z_hat"],
                              eps_pred=out["eps_pred"], eps=out["eps"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
        optimizer.step()

        pix_m.update(log["pix"].item())
        diff_m.update(log["diff"].item())

        if (it + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"[stage2][{it+1:>6d}/{args.total_iters}] "
                  f"lr={lr:.2e} pix={pix_m.avg:.4f} diff={diff_m.avg:.4f} "
                  f"({elapsed:.1f}s)")
            pix_m, diff_m = AverageMeter(), AverageMeter()
            t0 = time.time()

        if (it + 1) % args.val_every == 0 and val_loader is not None:
            vp, vs = validate(model, val_loader, device)
            print(f"[stage2][val] psnr={vp:.3f} ssim={vs:.4f}")

        if (it + 1) % args.ckpt_every == 0 or (it + 1) == args.total_iters:
            save_checkpoint(
                {"iter": it + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "args": vars(args)},
                os.path.join(args.out_dir, f"stage2_{it+1:06d}.pth"),
            )
            save_checkpoint(
                {"iter": it + 1, "model": model.state_dict()},
                os.path.join(args.out_dir, "stage2_latest.pth"),
            )

    print("[stage2] done.")


if __name__ == "__main__":
    main()
