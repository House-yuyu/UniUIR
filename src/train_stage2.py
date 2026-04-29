import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from .datasets import PairedUIRDataset
    from .losses import StageIILoss
    from .models import UniUIR, UniUIRConfig
    from .utils import (
        AverageMeter,
        cosine_lr,
        count_trainable_params,
        psnr,
        save_checkpoint,
        set_lr,
        ssim,
    )
except ImportError:
    from datasets import PairedUIRDataset
    from losses import StageIILoss
    from models import UniUIR, UniUIRConfig
    from utils import (
        AverageMeter,
        cosine_lr,
        count_trainable_params,
        psnr,
        save_checkpoint,
        set_lr,
        ssim,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--val_root", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="./checkpoints/stage2")
    p.add_argument("--stage1_ckpt", type=str, required=True,
                   help="Stage I checkpoint to initialize from")
    p.add_argument("--total_iters", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lambda_diff", type=float, default=1.0)
    p.add_argument("--lambda_eps", type=float, default=0.0,
                   help="optional weight for noise-prediction loss; 0 disables")
    p.add_argument("--depth_backend", type=str, default="hf",
                   choices=["hf", "dummy"])
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--val_every", type=int, default=5000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=1222)
    return p.parse_args()


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, device, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA devices.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    return True, device, rank, world_size


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def reduce_mean(value, device):
    if not is_dist_avail_and_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor.item()


def log_message(message, log_file=None):
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def cycle(loader, sampler=None, start_epoch=0):
    epoch = start_epoch
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    p_meter, s_meter = AverageMeter(), AverageMeter()
    for batch in loader:
        x_lq = batch["x_lq"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)
        # Use inference mode (LCDM sampling) at validation -- matches deployment.
        out = model(x_lq, mode="inference")
        x_hq = out["x_hq"].clamp(0, 1)
        p_meter.update(psnr(x_hq, x_gt), n=x_lq.size(0))
        s_meter.update(ssim(x_hq, x_gt), n=x_lq.size(0))
    model.train()
    model.depth_predictor.eval()
    model.sfpg.eval()
    return p_meter.avg, s_meter.avg


def main():
    args = parse_args()
    distributed, device, rank, world_size = setup_distributed()

    seed = args.seed + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log_file = None
    if is_main_process():
        log_path = Path(args.out_dir) / "stage2.log"
        log_file = open(log_path, "a", encoding="utf-8")
        log_message(f"[stage2] logging to {log_path}", log_file)

    train_set = PairedUIRDataset(args.data_root, crop_size=args.crop_size, augment=True)
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    train_iter = cycle(train_loader, sampler=train_sampler)

    val_loader = None
    if args.val_root and is_main_process():
        val_set = PairedUIRDataset(args.val_root, crop_size=args.crop_size,
                                   augment=False, resize_to=256)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    cfg = UniUIRConfig(depth_backend=args.depth_backend)
    model = UniUIR(cfg).to(device)
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if is_main_process():
        log_message(
            f"[stage2] loaded stage1: missing={len(missing)} unexpected={len(unexpected)}",
            log_file,
        )

    model.freeze_for_stage2()
    base_model = model
    if is_main_process():
        log_message(f"[stage2] trainable params (M): {count_trainable_params(base_model):.2f}",
                    log_file)

    train_params = (
        list(base_model.backbone.parameters())
        + list(base_model.sfpg_star.parameters())
        + list(base_model.lcdm.parameters())
    )
    optimizer = torch.optim.AdamW(
        train_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    criterion = StageIILoss(lambda_diff=args.lambda_diff,
                            lambda_eps=args.lambda_eps).to(device)

    start_iter = 0
    best_psnr = float("-inf")
    if args.resume and os.path.isfile(args.resume):
        rckpt = torch.load(args.resume, map_location="cpu")
        base_model.load_state_dict(rckpt["model"], strict=False)
        optimizer.load_state_dict(rckpt["optimizer"])
        start_iter = rckpt["iter"]
        best_psnr = rckpt.get("best_psnr", best_psnr)
        if is_main_process():
            log_message(f"[stage2] resumed from {args.resume} at iter {start_iter}", log_file)

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    pix_m, diff_m = AverageMeter(), AverageMeter()
    t0 = time.time()
    model.train()
    unwrap_model(model).depth_predictor.eval()
    unwrap_model(model).sfpg.eval()

    try:
        for it in range(start_iter, args.total_iters):
            lr = cosine_lr(it, args.total_iters, args.lr, args.min_lr)
            set_lr(optimizer, lr)

            batch = next(train_iter)
            x_lq = batch["x_lq"].to(device, non_blocking=True)
            x_gt = batch["x_gt"].to(device, non_blocking=True)

            out = model(x_lq, x_gt, mode="stage2")
            loss, log = criterion(
                out["x_hq"],
                x_gt,
                out["Z"],
                out["Z_hat"],
                eps_pred=out["eps_pred"],
                eps=out["eps"],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            optimizer.step()

            pix_value = reduce_mean(log["pix"].item(), device)
            diff_value = reduce_mean(log["diff"].item(), device)
            pix_m.update(pix_value)
            diff_m.update(diff_value)

            if (it + 1) % args.log_every == 0 and is_main_process():
                elapsed = time.time() - t0
                log_message(
                    f"[stage2][{it+1:>6d}/{args.total_iters}] "
                    f"lr={lr:.2e} pix={pix_m.avg:.4f} diff={diff_m.avg:.4f} "
                    f"({elapsed:.1f}s)",
                    log_file,
                )
                pix_m, diff_m = AverageMeter(), AverageMeter()
                t0 = time.time()

            if (it + 1) % args.val_every == 0 and args.val_root:
                if distributed:
                    dist.barrier()
                if val_loader is not None:
                    vp, vs = validate(unwrap_model(model), val_loader, device)
                    is_best = vp > best_psnr
                    if is_best:
                        best_psnr = vp
                    best_note = " best" if is_best else ""
                    log_message(f"[stage2][val] psnr={vp:.3f} ssim={vs:.4f}{best_note}",
                                log_file)
                    if is_best:
                        state_dict = unwrap_model(model).state_dict()
                        save_checkpoint(
                            {
                                "iter": it + 1,
                                "model": state_dict,
                                "optimizer": optimizer.state_dict(),
                                "args": vars(args),
                                "best_psnr": best_psnr,
                                "val_ssim": vs,
                            },
                            os.path.join(args.out_dir, "stage2_best.pth"),
                        )
                if distributed:
                    dist.barrier()
                model.train()
                unwrap_model(model).depth_predictor.eval()
                unwrap_model(model).sfpg.eval()

            if (it + 1) % args.ckpt_every == 0 or (it + 1) == args.total_iters:
                if is_main_process():
                    state_dict = unwrap_model(model).state_dict()
                    save_checkpoint(
                        {
                            "iter": it + 1,
                            "model": state_dict,
                            "optimizer": optimizer.state_dict(),
                            "args": vars(args),
                            "best_psnr": best_psnr,
                        },
                        os.path.join(args.out_dir, "stage2_latest.pth"),
                    )

        if is_main_process():
            log_message("[stage2] done.", log_file)
    finally:
        if log_file is not None:
            log_file.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
