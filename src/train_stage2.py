import argparse
import copy
import os
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
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
    p.add_argument("--out_dir", type=str, default="./checkpoints/stage2_paper")
    p.add_argument("--stage1_ckpt", type=str,
                   default='./checkpoints/stage1/stage1_best.pth',
                   help="Stage I checkpoint used to initialize backbone and SFPG")
    p.add_argument("--total_iters", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=128)
    p.add_argument(
        "--train_resize",
        type=int,
        default=0,
        help=(
            "resize each complete training pair to this square size before "
            "cropping; 0 keeps the original resolution"
        ),
    )
    p.add_argument(
        "--prior_resize",
        type=int,
        default=0,
        help=(
            "fixed full-image size used by SFPG/SFPG* during training; "
            "0 reuses the random restoration crop"
        ),
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--lcdm_lr_scale",
        type=float,
        default=1.0,
        help="LR multiplier for LCDM, independent of SFPG* and backbone",
    )
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--backbone_lr_scale", type=float, default=1.0,
                   help="LR multiplier for the Stage-I backbone during Stage-II")
    p.add_argument(
        "--prior_lr_scale",
        type=float,
        default=1.0,
        help="LR multiplier for SFPG*; 0 freezes the predicted-prior network",
    )
    p.add_argument(
        "--refiner_lr_scale",
        type=float,
        default=0.0,
        help="LR multiplier for the zero-initialized output residual refiner",
    )
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--prior_mode", type=str, default="diffusion",
                   choices=["direct", "diffusion"],
                   help="direct distills SFPG* to predict Z; diffusion uses LCDM sampling")
    p.add_argument(
        "--diffusion_target",
        choices=["teacher", "direct"],
        default="teacher",
        help=(
            "LCDM target: GT-conditioned frozen SFPG prior, or the strong "
            "inference-safe direct SFPG* prior"
        ),
    )
    p.add_argument(
        "--diffusion_train_start",
        choices=["terminal", "gaussian"],
        default="terminal",
        help=(
            "start reverse training from q(Z_T|Z) or pure Gaussian noise; "
            "gaussian exactly matches third-stage inference"
        ),
    )
    p.add_argument(
        "--diffusion_anchor_strength",
        type=float,
        default=1.0,
        help=(
            "blend from the stable direct SFPG* prior (0) to the raw LCDM "
            "sample (1); intermediate values train a condition-anchored LCDM"
        ),
    )
    p.add_argument(
        "--diffusion_stochastic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "inject noise between reverse steps; disabling it keeps the "
            "Gaussian initial state but makes the reverse trajectory deterministic"
        ),
    )
    p.add_argument(
        "--diffusion_state_clip",
        type=float,
        default=0.0,
        help=(
            "clamp normalized LCDM state after each reverse step; 0 disables "
            "clipping and 8 is a conservative stability setting"
        ),
    )
    p.add_argument(
        "--diffusion_eval_samples",
        type=int,
        default=1,
        help="number of Gaussian LCDM priors averaged before one validation backbone pass",
    )
    p.add_argument("--copy_sfpg_prompts", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="initialize SFPG* prompt bank from trained SFPG prompts")
    p.add_argument("--lambda_diff", type=float, default=1.0)
    p.add_argument(
        "--lambda_direct_diff",
        type=float,
        default=0.0,
        help="also distill the SFPG* direct condition sum toward the target prior",
    )
    p.add_argument(
        "--lambda_pix",
        type=float,
        default=1.0,
        help="weight of pixel L1; set to 0 for a pure PSNR-aligned MSE objective",
    )
    p.add_argument("--lambda_eps", type=float, default=0.0,
                   help="optional weight for noise-prediction loss; 0 disables")
    p.add_argument(
        "--lambda_mse",
        type=float,
        default=0.0,
        help="optional pixel MSE auxiliary loss, directly aligned with PSNR",
    )
    p.add_argument(
        "--lambda_log_mse",
        type=float,
        default=0.0,
        help=(
            "weight of mean per-image log(MSE), which is directly equivalent "
            "to maximizing mean per-image PSNR"
        ),
    )
    p.add_argument(
        "--lambda_teacher",
        type=float,
        default=0.0,
        help=(
            "image-level distillation weight from a frozen Stage-I backbone; "
            "0 disables teacher consistency"
        ),
    )
    p.add_argument(
        "--lambda_teacher_mse",
        type=float,
        default=0.0,
        help="optional MSE distillation from the frozen Stage-I image teacher",
    )
    p.add_argument(
        "--latent_scale",
        type=float,
        default=64.0,
        help="scale SFPG latents to approximately unit variance for diffusion",
    )
    p.add_argument(
        "--prior_warmup_iters",
        "--teacher_warmup_iters",
        dest="prior_warmup_iters",
        type=int,
        default=0,
        help=(
            "train only SFPG*/LCDM before deployment-path joint fine-tuning; "
            "--teacher_warmup_iters is accepted as a deprecated alias"
        ),
    )
    p.add_argument("--val_teacher", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="also validate with frozen SFPG(x_lq,x_gt) teacher prior")
    p.add_argument(
        "--sanity_val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate Stage-I teacher path once before Stage-II training",
    )
    p.add_argument("--val_direct", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="also validate the direct SFPG* prior path for debugging")
    p.add_argument("--depth_backend", type=str, default="hf",
                   choices=["hf", "dummy"])
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=500) # 500
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--reset_backbone_on_resume",
        action="store_true",
        help=(
            "keep resumed SFPG*/LCDM weights but restore the Stage-I backbone; "
            "useful when an earlier joint run damaged the restoration backbone"
        ),
    )
    p.add_argument(
        "--reset_optimizer_on_resume",
        action="store_true",
        help="resume model/iteration but start with a fresh optimizer",
    )
    p.add_argument(
        "--reset_iter_on_resume",
        action="store_true",
        help=(
            "load checkpoint weights but restart the iteration/LR schedule "
            "from zero and reset the tracked best metric"
        ),
    )
    p.add_argument(
        "--reset_lcdm_on_resume",
        action="store_true",
        help="load resumed backbone/SFPG*/refiner but reinitialize LCDM",
    )
    p.add_argument("--allow_legacy_resume", action="store_true",
                   help="allow resuming a checkpoint made by the old broken Stage-II loader")
    p.add_argument("--seed", type=int, default=1222)
    p.add_argument("--val_seed", type=int, default=2025,
                   help="fixed seed for deterministic validation sampling")
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="evaluate resumed deployment/direct/teacher paths and exit",
    )
    p.add_argument(
        "--eval_blend_sweep",
        action="store_true",
        help="during eval, scan residual output strength for PSNR calibration",
    )
    p.add_argument(
        "--eval_prior_blend_sweep",
        action="store_true",
        help="evaluate direct-to-LCDM prior anchor strengths with fixed noise",
    )
    p.add_argument(
        "--eval_tta",
        action="store_true",
        help="during eval, also test 4-way flip test-time augmentation",
    )
    p.add_argument(
        "--eval_deployment_only",
        action="store_true",
        help="skip auxiliary direct/self/teacher paths during eval",
    )
    p.add_argument(
        "--eval_ensemble_resume",
        type=str,
        default=None,
        help="second Stage-II checkpoint used for an output-ensemble PSNR sweep",
    )
    p.add_argument(
        "--paper_stage2",
        action="store_true",
        help=(
            "enforce the paper Stage-II recipe: four-step diffusion, joint "
            "fine-tuning, 128 crops, L1 image + raw-latent L1 only"
        ),
    )
    args = p.parse_args()
    if not 0.0 <= args.diffusion_anchor_strength <= 1.0:
        p.error("--diffusion_anchor_strength must be between 0 and 1")
    if args.diffusion_state_clip < 0:
        p.error("--diffusion_state_clip must be non-negative")
    if args.lcdm_lr_scale < 0:
        p.error("--lcdm_lr_scale must be non-negative")
    if args.max_grad_norm <= 0:
        p.error("--max_grad_norm must be positive")
    if args.diffusion_eval_samples < 1:
        p.error("--diffusion_eval_samples must be at least 1")
    if args.paper_stage2:
        args.total_iters = 200000
        args.crop_size = 128
        args.train_resize = 0
        args.prior_resize = 0
        args.lr = 2e-4
        args.min_lr = 1e-6
        args.backbone_lr_scale = 1.0
        args.prior_lr_scale = 1.0
        args.refiner_lr_scale = 0.0
        args.prior_mode = "diffusion"
        args.diffusion_target = "teacher"
        args.diffusion_train_start = "terminal"
        args.diffusion_anchor_strength = 1.0
        args.diffusion_stochastic = True
        args.diffusion_state_clip = 0.0
        args.lambda_diff = 1.0
        args.lambda_direct_diff = 0.0
        args.lambda_pix = 1.0
        args.lambda_eps = 0.0
        args.lambda_mse = 0.0
        args.lambda_log_mse = 0.0
        args.lambda_teacher = 0.0
        args.lambda_teacher_mse = 0.0
        args.prior_warmup_iters = 0
    return args


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


def format_duration(seconds):
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def count_params(module, trainable_only=False):
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params) / 1e6


def log_module_param_counts(model, log_file=None):
    modules = [
        ("MMoE-UIR/backbone", model.backbone),
        ("SFPG teacher", model.sfpg),
        ("SFPG*", model.sfpg_star),
        ("LCDM", model.lcdm),
        ("output refiner", model.output_refiner),
        ("depth predictor", model.depth_predictor),
    ]
    for name, module in modules:
        log_message(
            f"[stage2][params] {name}: "
            f"trainable={count_params(module, True):.2f}M/"
            f"total={count_params(module, False):.2f}M",
            log_file,
        )


def load_stage1_for_stage2(model, state_dict):
    """Initialize the Stage-II backbone and frozen SFPG from Stage I."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value

    stage1_state = {
        key: value
        for key, value in normalized.items()
        if key.startswith("backbone.") or key.startswith("sfpg.")
    }
    backbone_keys = sum(key.startswith("backbone.") for key in stage1_state)
    sfpg_keys = sum(key.startswith("sfpg.") for key in stage1_state)
    if not backbone_keys or not sfpg_keys:
        raise RuntimeError(
            "Stage I checkpoint must contain both backbone.* and sfpg.* weights; "
            f"found backbone={backbone_keys}, sfpg={sfpg_keys}."
        )

    missing, unexpected = model.load_state_dict(stage1_state, strict=False)
    relevant_missing = [
        key for key in missing
        if key.startswith("backbone.") or key.startswith("sfpg.")
    ]
    if relevant_missing or unexpected:
        raise RuntimeError(
            "Failed to initialize Stage II from Stage I: "
            f"missing={relevant_missing}, unexpected={unexpected}"
        )
    return backbone_keys, sfpg_keys


def initialize_sfpg_star_from_sfpg(model):
    """Initialize SFPG* from the trained Stage-I SFPG.

    SFPG* differs only in its input stem: SFPG receives two RGB images after
    4x PixelUnshuffle (96 channels), whereas SFPG* receives one RGB image.
    All frequency/residual/tail weights are directly reusable. For the stem,
    fold the 16 sub-pixel filters and both RGB input slots into three channels.
    This gives Stage II a trained condition encoder instead of a random one.
    """
    source = model.sfpg.net
    target = model.sfpg_star.net

    target.amp_branch.load_state_dict(source.amp_branch.state_dict())
    target.phase_branch.load_state_dict(source.phase_branch.state_dict())
    target.tail.load_state_dict(source.tail.state_dict())
    # Reuse the trained feature refinement in the richer SFPG* condition head;
    # its final N*C_hat projection intentionally remains zero-initialized.
    model.sfpg_star.condition_head[0].load_state_dict(source.tail[0].state_dict())
    model.sfpg_star.condition_head[2].load_state_dict(source.tail[2].state_dict())

    source_conv = source.head[1]
    target_conv = target.head[0]
    if source_conv.in_channels != 6 * 16 or target_conv.in_channels != 3:
        raise RuntimeError(
            "Unexpected SFPG/SFPG* stem shapes: "
            f"source={source_conv.weight.shape}, target={target_conv.weight.shape}"
        )
    with torch.no_grad():
        weight = source_conv.weight.view(
            source_conv.out_channels,
            6,
            16,
            *source_conv.kernel_size,
        )
        # If both Stage-I input slots contained the same slowly varying RGB
        # image, this sum exactly preserves the stem response.
        folded = weight[:, :3].sum(dim=2) + weight[:, 3:].sum(dim=2)
        target_conv.weight.copy_(folded)
        if source_conv.bias is not None:
            target_conv.bias.copy_(source_conv.bias)
        target.prompts.copy_(source.prompts)


def stage2_state_dict(model):
    """Stage II checkpoint stores only deployment/trainable modules.

    The frozen SFPG teacher is reloaded from Stage I, and Depth Anything is a
    frozen external model, so saving them only bloats checkpoints.
    """
    state = model.state_dict()
    return {
        key: value
        for key, value in state.items()
        if (
            key.startswith("backbone.")
            or key.startswith("sfpg_star.")
            or key.startswith("lcdm.")
            or key.startswith("output_refiner.")
        )
    }


def cycle(loader, sampler=None, start_epoch=0):
    epoch = start_epoch
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def make_generator(device, seed):
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    prior_mode="direct",
    val_seed=2025,
    diffusion_anchor_strength=1.0,
    diffusion_stochastic=True,
    diffusion_samples=1,
):
    model.eval()
    p_meter, s_meter = AverageMeter(), AverageMeter()
    for idx, batch in enumerate(loader):
        x_lq = batch["x_lq"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)
        # Use inference mode (LCDM sampling) at validation -- matches deployment.
        generator = None
        if prior_mode == "diffusion":
            generator = make_generator(device, val_seed + idx)
        out = model(
            x_lq,
            mode="inference",
            prior_mode=prior_mode,
            sample_generator=generator,
            diffusion_anchor_strength=diffusion_anchor_strength,
            diffusion_stochastic=diffusion_stochastic,
            diffusion_samples=diffusion_samples,
        )
        x_hq = out["x_hq"].clamp(0, 1)
        p_meter.update(psnr(x_hq, x_gt), n=x_lq.size(0))
        s_meter.update(ssim(x_hq, x_gt), n=x_lq.size(0))
    model.train()
    model.depth_predictor.eval()
    model.sfpg.eval()
    return p_meter.avg, s_meter.avg


@torch.no_grad()
def validate_prior_blend_sweep(
    model,
    loader,
    device,
    val_seed=2025,
    diffusion_stochastic=True,
    diffusion_samples=1,
):
    """Measure condition-anchor strengths on the exact deployment path."""
    strengths = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
    results = []
    for strength in strengths:
        vp, vs = validate(
            model,
            loader,
            device,
            prior_mode="diffusion",
            val_seed=val_seed,
            diffusion_anchor_strength=strength,
            diffusion_stochastic=diffusion_stochastic,
            diffusion_samples=diffusion_samples,
        )
        results.append((strength, vp, vs))
    return results


@torch.no_grad()
def validate_flip_tta(
    model,
    loader,
    device,
    prior_mode="direct",
    val_seed=2025,
    diffusion_anchor_strength=1.0,
    diffusion_stochastic=True,
    diffusion_samples=1,
):
    """Validate an original/horizontal/vertical/both flip output ensemble."""
    model.eval()
    p_meter, s_meter = AverageMeter(), AverageMeter()
    flip_dims = [(), (-1,), (-2,), (-2, -1)]
    for idx, batch in enumerate(loader):
        x_lq = batch["x_lq"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)
        predictions = []
        for variant_index, dims in enumerate(flip_dims):
            transformed = torch.flip(x_lq, dims=dims) if dims else x_lq
            generator = None
            if prior_mode == "diffusion":
                generator = make_generator(
                    device,
                    val_seed + idx * len(flip_dims) + variant_index,
                )
            prediction = model(
                transformed,
                mode="inference",
                prior_mode=prior_mode,
                sample_generator=generator,
                diffusion_anchor_strength=diffusion_anchor_strength,
                diffusion_stochastic=diffusion_stochastic,
                diffusion_samples=diffusion_samples,
            )["x_hq"]
            if dims:
                prediction = torch.flip(prediction, dims=dims)
            predictions.append(prediction)
        x_hq = torch.stack(predictions, dim=0).mean(dim=0).clamp(0, 1)
        p_meter.update(psnr(x_hq, x_gt), n=x_lq.size(0))
        s_meter.update(ssim(x_hq, x_gt), n=x_lq.size(0))
    model.train()
    model.depth_predictor.eval()
    model.sfpg.eval()
    return p_meter.avg, s_meter.avg


@torch.no_grad()
def validate_teacher(model, loader, device):
    model.eval()
    p_meter, s_meter = AverageMeter(), AverageMeter()
    for batch in loader:
        x_lq = batch["x_lq"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)
        out = model(x_lq, x_gt, mode="teacher")
        x_hq = out["x_hq"].clamp(0, 1)
        p_meter.update(psnr(x_hq, x_gt), n=x_lq.size(0))
        s_meter.update(ssim(x_hq, x_gt), n=x_lq.size(0))
    model.train()
    model.depth_predictor.eval()
    model.sfpg.eval()
    return p_meter.avg, s_meter.avg


@torch.no_grad()
def validate_blend_sweep(model, loader, device, prior_mode="direct"):
    """Find the output residual strength best aligned with validation PSNR."""
    model.eval()
    alphas = [0.70 + 0.025 * i for i in range(25)]
    meters = [AverageMeter() for _ in alphas]
    for batch in loader:
        x_lq = batch["x_lq"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)
        out = model(x_lq, mode="inference", prior_mode=prior_mode)
        residual = out["x_hq"] - x_lq
        for alpha, meter in zip(alphas, meters):
            calibrated = (x_lq + alpha * residual).clamp(0, 1)
            meter.update(psnr(calibrated, x_gt), n=x_lq.size(0))
    model.train()
    model.depth_predictor.eval()
    model.sfpg.eval()
    return [(alpha, meter.avg) for alpha, meter in zip(alphas, meters)]


@torch.no_grad()
def validate_checkpoint_ensemble(
    model,
    loader,
    device,
    second_checkpoint,
    prior_mode="direct",
):
    """Sweep output averaging between the loaded model and another checkpoint."""
    model.eval()
    first_outputs = []
    for batch in loader:
        x_lq = batch["x_lq"].to(device, non_blocking=True)
        out = model(x_lq, mode="inference", prior_mode=prior_mode)
        first_outputs.append(out["x_hq"].clamp(0, 1).cpu())

    second = torch.load(second_checkpoint, map_location="cpu")
    second_state = second["model"] if "model" in second else second
    normalized = {}
    for key, value in second_state.items():
        normalized[key[len("module."):] if key.startswith("module.") else key] = value
    missing, unexpected = model.load_state_dict(normalized, strict=False)
    required_missing = [
        key
        for key in missing
        if (
            key.startswith("backbone.")
            or key.startswith("sfpg_star.")
            or key.startswith("lcdm.")
        )
    ]
    if required_missing or unexpected:
        raise RuntimeError(
            "Failed to load ensemble checkpoint: "
            f"missing={required_missing}, unexpected={unexpected}"
        )

    alphas = [0.025 * i for i in range(41)]
    meters = [AverageMeter() for _ in alphas]
    for first, batch in zip(first_outputs, loader):
        x_lq = batch["x_lq"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)
        second_out = model(
            x_lq,
            mode="inference",
            prior_mode=prior_mode,
        )["x_hq"].clamp(0, 1)
        first = first.to(device)
        for alpha, meter in zip(alphas, meters):
            ensemble = (1.0 - alpha) * first + alpha * second_out
            meter.update(psnr(ensemble, x_gt), n=x_lq.size(0))
    return [(alpha, meter.avg) for alpha, meter in zip(alphas, meters)]


def main():
    args = parse_args()
    distributed, device, rank, world_size = setup_distributed()

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log_file = None
    if is_main_process():
        run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(args.out_dir) / f"stage2_{run_time}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        log_message(f"[stage2] logging to {log_path}", log_file)
        log_message(f"[stage2] run_time={run_time}", log_file)

    train_set = PairedUIRDataset(
        args.data_root,
        crop_size=args.crop_size,
        augment=True,
        resize_to=args.train_resize if args.train_resize > 0 else None,
        prior_resize_to=args.prior_resize if args.prior_resize > 0 else None,
    )
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
        # Reuse the requested worker count so constrained launch environments
        # can run both training and validation with ``--num_workers 0``.
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

    cfg = UniUIRConfig(
        depth_backend=args.depth_backend,
        latent_scale=args.latent_scale,
        diffusion_state_clip=args.diffusion_state_clip,
    )
    model = UniUIR(cfg).to(device)
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    loaded_backbone_keys, loaded_sfpg_keys = load_stage1_for_stage2(model, sd)
    teacher_backbone = None
    if args.lambda_teacher > 0 or args.lambda_teacher_mse > 0:
        # Keep a pristine Stage-I image teacher before a resumed Stage-II
        # checkpoint can overwrite the student backbone.
        teacher_backbone = copy.deepcopy(model.backbone).to(device).eval()
        for parameter in teacher_backbone.parameters():
            parameter.requires_grad_(False)
    if args.copy_sfpg_prompts:
        initialize_sfpg_star_from_sfpg(model)
    if is_main_process():
        log_message(
            "[stage2] initialized from Stage I: "
            f"backbone_keys={loaded_backbone_keys} sfpg_keys={loaded_sfpg_keys}",
            log_file,
        )
        if args.paper_stage2:
            log_message(
                "[stage2] recipe=paper: joint four-step diffusion, "
                "L1(image)+L1(raw latent), no warmup/teacher/MSE/eps auxiliary",
                log_file,
            )
        if args.copy_sfpg_prompts:
            log_message(
                "[stage2] initialized complete SFPG* encoder from trained SFPG "
                "(folded input stem + copied frequency/tail/prompt weights)",
                log_file,
            )

    model.freeze_for_stage2()
    if args.prior_mode == "direct":
        for p in model.lcdm.parameters():
            p.requires_grad_(False)
        model.lcdm.eval()
    elif args.lcdm_lr_scale <= 0:
        for p in model.lcdm.parameters():
            p.requires_grad_(False)
        model.lcdm.eval()
    if args.backbone_lr_scale <= 0:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        model.backbone.eval()
    if args.prior_lr_scale <= 0:
        for p in model.sfpg_star.parameters():
            p.requires_grad_(False)
        model.sfpg_star.eval()
    if args.refiner_lr_scale <= 0:
        for p in model.output_refiner.parameters():
            p.requires_grad_(False)
        model.output_refiner.eval()
    base_model = model
    if is_main_process():
        log_message(
            f"[stage2] prior_mode={args.prior_mode} "
            f"diffusion_target={args.diffusion_target} "
            f"diffusion_train_start={args.diffusion_train_start} "
            f"diffusion_anchor_strength={args.diffusion_anchor_strength} "
            f"diffusion_stochastic={args.diffusion_stochastic} "
            f"diffusion_state_clip={args.diffusion_state_clip} "
            f"diffusion_eval_samples={args.diffusion_eval_samples} "
            f"backbone_lr_scale={args.backbone_lr_scale} "
            f"prior_warmup_iters={args.prior_warmup_iters} "
            f"lambda_pix={args.lambda_pix} "
            f"lambda_diff={args.lambda_diff} "
            f"lambda_direct_diff={args.lambda_direct_diff} "
            f"lambda_mse={args.lambda_mse} "
            f"lambda_log_mse={args.lambda_log_mse} "
            f"lambda_teacher={args.lambda_teacher} "
            f"lambda_teacher_mse={args.lambda_teacher_mse} "
            f"refiner_lr_scale={args.refiner_lr_scale} "
            f"train_resize={args.train_resize}",
            log_file,
        )
        log_module_param_counts(base_model, log_file)
        log_message(f"[stage2] trainable params (M): {count_trainable_params(base_model):.2f}",
                    log_file)

    param_groups = []
    if args.backbone_lr_scale > 0:
        backbone_params = [p for p in base_model.backbone.parameters() if p.requires_grad]
        param_groups.append({
            "params": backbone_params,
            "lr_scale": args.backbone_lr_scale,
        })
    if args.prior_lr_scale > 0:
        sfpg_star_params = [
            p for p in base_model.sfpg_star.parameters() if p.requires_grad
        ]
        param_groups.append({
            "params": sfpg_star_params,
            "lr_scale": args.prior_lr_scale,
        })
    if args.prior_mode == "diffusion" and args.lcdm_lr_scale > 0:
        lcdm_params = [p for p in base_model.lcdm.parameters() if p.requires_grad]
        param_groups.append({
            "params": lcdm_params,
            "lr_scale": args.lcdm_lr_scale,
        })
    if args.refiner_lr_scale > 0:
        refiner_params = [
            p for p in base_model.output_refiner.parameters()
            if p.requires_grad
        ]
        param_groups.append({
            "params": refiner_params,
            "lr_scale": args.refiner_lr_scale,
        })
    train_params = [
        p for group in param_groups
        for p in group["params"]
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    criterion = StageIILoss(
        lambda_pix=args.lambda_pix,
        lambda_diff=args.lambda_diff,
        lambda_eps=args.lambda_eps,
        lambda_teacher=args.lambda_teacher,
        lambda_teacher_mse=args.lambda_teacher_mse,
        lambda_mse=args.lambda_mse,
        lambda_log_mse=args.lambda_log_mse,
        lambda_direct_diff=args.lambda_direct_diff,
        latent_scale=args.latent_scale,
    ).to(device)

    start_iter = 0
    best_psnr = float("-inf")
    if args.resume and os.path.isfile(args.resume):
        fresh_lcdm_state = None
        if args.reset_lcdm_on_resume:
            fresh_lcdm_state = copy.deepcopy(base_model.lcdm.state_dict())
        rckpt = torch.load(args.resume, map_location="cpu")
        training_version = rckpt.get("training_version", 1)
        if training_version < 3 and not args.allow_legacy_resume:
            raise RuntimeError(
                "This checkpoint predates the latent-normalized corrected "
                "Stage-II pipeline. Restart Stage II, or pass "
                "--allow_legacy_resume only if intentional."
            )
        resume_missing, resume_unexpected = base_model.load_state_dict(
            rckpt["model"],
            strict=False,
        )
        required_prefixes = ("backbone.", "sfpg_star.", "lcdm.")
        required_missing = [
            key for key in resume_missing
            if key.startswith(required_prefixes)
        ]
        if required_missing or resume_unexpected:
            raise RuntimeError(
                "Failed to load Stage-II resume checkpoint: "
                f"missing={required_missing}, unexpected={resume_unexpected}"
            )
        if fresh_lcdm_state is not None:
            base_model.lcdm.load_state_dict(fresh_lcdm_state)
            if is_main_process():
                log_message(
                    "[stage2] reinitialized LCDM after loading resumed "
                    "backbone/SFPG*/refiner",
                    log_file,
                )
        if args.reset_backbone_on_resume:
            load_stage1_for_stage2(base_model, sd)
            if is_main_process():
                log_message(
                    "[stage2] restored Stage-I backbone after loading resumed "
                    "SFPG*/LCDM weights",
                    log_file,
                )
        if (
            not args.reset_optimizer_on_resume
            and not args.reset_iter_on_resume
            and "optimizer" in rckpt
        ):
            try:
                optimizer.load_state_dict(rckpt["optimizer"])
                # Optimizer state dictionaries also store custom param-group
                # fields. Re-apply the current CLI LR scales so an old resume
                # cannot silently override a changed backbone_lr_scale.
                group_index = 0
                if args.backbone_lr_scale > 0:
                    optimizer.param_groups[group_index]["lr_scale"] = (
                        args.backbone_lr_scale
                    )
                    group_index += 1
                if args.prior_lr_scale > 0:
                    optimizer.param_groups[group_index]["lr_scale"] = (
                        args.prior_lr_scale
                    )
                    group_index += 1
                if (
                    args.prior_mode == "diffusion"
                    and args.lcdm_lr_scale > 0
                ):
                    optimizer.param_groups[group_index]["lr_scale"] = (
                        args.lcdm_lr_scale
                    )
                    group_index += 1
                if args.refiner_lr_scale > 0:
                    optimizer.param_groups[group_index]["lr_scale"] = (
                        args.refiner_lr_scale
                    )
            except ValueError as exc:
                if is_main_process():
                    log_message(
                        f"[stage2] optimizer state not loaded: {exc}; "
                        "continuing with fresh optimizer",
                        log_file,
                    )
        elif is_main_process():
            log_message(
                "[stage2] optimizer state reset by request or absent from checkpoint",
                log_file,
            )
        start_iter = 0 if args.reset_iter_on_resume else rckpt["iter"]
        if (
            args.reset_backbone_on_resume
            or args.reset_lcdm_on_resume
            or args.reset_iter_on_resume
        ):
            best_psnr = float("-inf")
        else:
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

    if args.eval_only:
        if not args.val_root:
            raise ValueError("--eval_only requires --val_root")
        if distributed:
            dist.barrier()
        if val_loader is not None:
            eval_model = unwrap_model(model)
            vp, vs = validate(
                eval_model,
                val_loader,
                device,
                prior_mode=args.prior_mode,
                val_seed=args.val_seed,
                diffusion_anchor_strength=args.diffusion_anchor_strength,
                diffusion_stochastic=args.diffusion_stochastic,
                diffusion_samples=args.diffusion_eval_samples,
            )
            log_message(
                f"[stage2][eval] deployment psnr={vp:.3f} ssim={vs:.4f}",
                log_file,
            )
            if args.eval_tta:
                tta_p, tta_s = validate_flip_tta(
                    eval_model,
                    val_loader,
                    device,
                    prior_mode=args.prior_mode,
                    val_seed=args.val_seed,
                    diffusion_anchor_strength=args.diffusion_anchor_strength,
                    diffusion_stochastic=args.diffusion_stochastic,
                    diffusion_samples=args.diffusion_eval_samples,
                )
                log_message(
                    f"[stage2][eval][tta] psnr={tta_p:.3f} "
                    f"ssim={tta_s:.4f}",
                    log_file,
                )
            if args.eval_ensemble_resume:
                ensemble_sweep = validate_checkpoint_ensemble(
                    eval_model,
                    val_loader,
                    device,
                    second_checkpoint=args.eval_ensemble_resume,
                    prior_mode=args.prior_mode,
                )
                ensemble_alpha, ensemble_psnr = max(
                    ensemble_sweep,
                    key=lambda item: item[1],
                )
                log_message(
                    f"[stage2][eval][ensemble] alpha={ensemble_alpha:.3f} "
                    f"psnr={ensemble_psnr:.3f}",
                    log_file,
                )
            if not args.eval_deployment_only:
                direct_p, direct_s = validate(
                    eval_model,
                    val_loader,
                    device,
                    prior_mode="direct",
                    val_seed=args.val_seed,
                )
                self_p, self_s = validate(
                    eval_model,
                    val_loader,
                    device,
                    prior_mode="self",
                    val_seed=args.val_seed,
                )
                teacher_p, teacher_s = validate_teacher(
                    eval_model,
                    val_loader,
                    device,
                )
                log_message(
                    f"[stage2][eval][direct] psnr={direct_p:.3f} "
                    f"ssim={direct_s:.4f}",
                    log_file,
                )
                log_message(
                    f"[stage2][eval][self] psnr={self_p:.3f} "
                    f"ssim={self_s:.4f}",
                    log_file,
                )
                log_message(
                    f"[stage2][eval][teacher] psnr={teacher_p:.3f} "
                    f"ssim={teacher_s:.4f}",
                    log_file,
                )
            if args.eval_blend_sweep:
                sweep = validate_blend_sweep(
                    eval_model,
                    val_loader,
                    device,
                    prior_mode=args.prior_mode,
                )
                best_alpha, best_blend_psnr = max(sweep, key=lambda item: item[1])
                log_message(
                    f"[stage2][eval][blend] alpha={best_alpha:.3f} "
                    f"psnr={best_blend_psnr:.3f}",
                    log_file,
                )
            if args.eval_prior_blend_sweep:
                prior_sweep = validate_prior_blend_sweep(
                    eval_model,
                    val_loader,
                    device,
                    val_seed=args.val_seed,
                    diffusion_stochastic=args.diffusion_stochastic,
                )
                for strength, blend_p, blend_s in prior_sweep:
                    log_message(
                        f"[stage2][eval][prior_blend] "
                        f"strength={strength:.2f} psnr={blend_p:.3f} "
                        f"ssim={blend_s:.4f}",
                        log_file,
                    )
        if distributed:
            dist.barrier()
        if log_file is not None:
            log_file.close()
        cleanup_distributed()
        return

    if args.sanity_val and args.val_root:
        if distributed:
            dist.barrier()
        if val_loader is not None:
            init_teacher_p, init_teacher_s = validate_teacher(
                unwrap_model(model),
                val_loader,
                device,
            )
            sanity_note = (
                "; this should reproduce Stage I"
                if start_iter == 0 or args.reset_backbone_on_resume
                else "; resumed-backbone diagnostic"
            )
            log_message(
                f"[stage2][sanity][teacher] psnr={init_teacher_p:.3f} "
                f"ssim={init_teacher_s:.4f}{sanity_note}",
                log_file,
            )
        if distributed:
            dist.barrier()

    pix_m = AverageMeter()
    mse_m = AverageMeter()
    log_mse_m = AverageMeter()
    diff_m = AverageMeter()
    eps_m = AverageMeter()
    teacher_m = AverageMeter()
    teacher_mse_m = AverageMeter()
    train_start_time = time.time()
    t0 = time.time()
    model.train()
    unwrap_model(model).depth_predictor.eval()
    unwrap_model(model).sfpg.eval()
    if args.prior_mode == "direct":
        unwrap_model(model).lcdm.eval()
    elif args.lcdm_lr_scale <= 0:
        unwrap_model(model).lcdm.eval()
    if args.backbone_lr_scale <= 0:
        unwrap_model(model).backbone.eval()
    if args.prior_lr_scale <= 0:
        unwrap_model(model).sfpg_star.eval()
    if args.refiner_lr_scale <= 0:
        unwrap_model(model).output_refiner.eval()

    try:
        for it in range(start_iter, args.total_iters):
            lr = cosine_lr(it, args.total_iters, args.lr, args.min_lr)
            set_lr(optimizer, lr)

            batch = next(train_iter)
            x_lq = batch["x_lq"].to(device, non_blocking=True)
            x_gt = batch["x_gt"].to(device, non_blocking=True)
            prior_lq = batch.get("prior_lq")
            prior_gt = batch.get("prior_gt")
            if prior_lq is not None:
                prior_lq = prior_lq.to(device, non_blocking=True)
                prior_gt = prior_gt.to(device, non_blocking=True)

            in_prior_warmup = it < args.prior_warmup_iters
            if in_prior_warmup:
                out = model(
                    x_lq,
                    x_gt,
                    mode="prior_distill",
                    prior_mode=args.prior_mode,
                    prior_lq=prior_lq,
                    prior_gt=prior_gt,
                    compute_eps_aux=args.lambda_eps > 0,
                    diffusion_target=args.diffusion_target,
                    diffusion_train_start=args.diffusion_train_start,
                    diffusion_anchor_strength=args.diffusion_anchor_strength,
                    diffusion_stochastic=args.diffusion_stochastic,
                )
                loss, log = criterion.prior_loss(
                    out["Z"],
                    out["Z_diffusion"],
                    eps_pred=out["eps_pred"],
                    eps=out["eps"],
                    Z_direct=out["Z_direct"],
                )
            else:
                out = model(
                    x_lq,
                    x_gt,
                    mode="stage2",
                    prior_mode=args.prior_mode,
                    prior_lq=prior_lq,
                    prior_gt=prior_gt,
                    compute_eps_aux=args.lambda_eps > 0,
                    diffusion_target=args.diffusion_target,
                    diffusion_train_start=args.diffusion_train_start,
                    diffusion_anchor_strength=args.diffusion_anchor_strength,
                    diffusion_stochastic=args.diffusion_stochastic,
                )
                x_teacher = None
                if teacher_backbone is not None:
                    with torch.no_grad():
                        x_teacher = teacher_backbone(
                            x_lq,
                            prior=out["Z"],
                            depth=out["depth"],
                        )
                loss, log = criterion(
                    out["x_hq"],
                    x_gt,
                    out["Z"],
                    out["Z_diffusion"],
                    eps_pred=out["eps_pred"],
                    eps=out["eps"],
                    x_teacher=x_teacher,
                    Z_direct=out["Z_direct"],
                )

            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                message = (
                    f"non-finite loss at iteration {it + 1}; aborting before "
                    "the optimizer can corrupt the last finite checkpoint"
                )
                log_message(f"[stage2][error] {message}", log_file)
                raise FloatingPointError(message)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_params,
                max_norm=args.max_grad_norm,
            )
            if not torch.isfinite(grad_norm):
                message = (
                    f"non-finite gradient norm at iteration {it + 1}; "
                    "aborting before optimizer.step()"
                )
                log_message(f"[stage2][error] {message}", log_file)
                raise FloatingPointError(message)
            optimizer.step()

            pix_value = None
            if "pix" in log:
                pix_value = reduce_mean(log["pix"].item(), device)
            diff_value = reduce_mean(log["diff"].item(), device)
            if pix_value is not None:
                pix_m.update(pix_value)
            if "mse" in log:
                mse_value = reduce_mean(log["mse"].item(), device)
                mse_m.update(mse_value)
            if "log_mse" in log:
                log_mse_value = reduce_mean(log["log_mse"].item(), device)
                log_mse_m.update(log_mse_value)
            diff_m.update(diff_value)
            if "eps" in log:
                eps_value = reduce_mean(log["eps"].item(), device)
                eps_m.update(eps_value)
            if "teacher" in log:
                teacher_value = reduce_mean(log["teacher"].item(), device)
                teacher_m.update(teacher_value)
            if "teacher_mse" in log:
                teacher_mse_value = reduce_mean(
                    log["teacher_mse"].item(),
                    device,
                )
                teacher_mse_m.update(teacher_mse_value)

            if (it + 1) % args.log_every == 0 and is_main_process():
                elapsed = time.time() - t0
                done_iters = (it + 1) - start_iter
                avg_iter_time = (time.time() - train_start_time) / max(1, done_iters)
                eta = avg_iter_time * (args.total_iters - (it + 1))
                bb_lr = lr * max(args.backbone_lr_scale, 0.0)
                lcdm_lr = lr * max(args.lcdm_lr_scale, 0.0)
                refiner_lr = lr * max(args.refiner_lr_scale, 0.0)
                eps_text = f" eps={eps_m.avg:.4f}" if eps_m.count else ""
                teacher_text = (
                    f" teacher={teacher_m.avg:.4f}"
                    if teacher_m.count
                    else ""
                )
                teacher_mse_text = (
                    f" teacher_mse={teacher_mse_m.avg:.5f}"
                    if teacher_mse_m.count
                    else ""
                )
                pix_text = f"pix={pix_m.avg:.4f} " if pix_m.count else ""
                mse_text = f"mse={mse_m.avg:.5f} " if mse_m.count else ""
                log_mse_text = (
                    f"log_mse={log_mse_m.avg:.4f} "
                    if log_mse_m.count
                    else ""
                )
                phase = "prior_warmup" if in_prior_warmup else args.prior_mode
                log_message(
                    f"[stage2][{it+1:>6d}/{args.total_iters}] "
                    f"phase={phase} lr={lr:.2e} bb_lr={bb_lr:.2e} "
                    f"lcdm_lr={lcdm_lr:.2e} "
                    f"refiner_lr={refiner_lr:.2e} "
                    f"{pix_text}{mse_text}{log_mse_text}"
                    f"diff={diff_m.avg:.4f}{eps_text}"
                    f"{teacher_text}{teacher_mse_text} "
                    f"({elapsed:.1f}s) eta={format_duration(eta)}",
                    log_file,
                )
                pix_m = AverageMeter()
                mse_m = AverageMeter()
                log_mse_m = AverageMeter()
                diff_m = AverageMeter()
                eps_m = AverageMeter()
                teacher_m = AverageMeter()
                teacher_mse_m = AverageMeter()
                t0 = time.time()

            if (it + 1) % args.val_every == 0 and args.val_root:
                if distributed:
                    dist.barrier()
                if val_loader is not None:
                    vp, vs = validate(
                        unwrap_model(model),
                        val_loader,
                        device,
                        prior_mode=args.prior_mode,
                        val_seed=args.val_seed,
                        diffusion_anchor_strength=args.diffusion_anchor_strength,
                        diffusion_stochastic=args.diffusion_stochastic,
                        diffusion_samples=args.diffusion_eval_samples,
                    )
                    is_best = vp > best_psnr
                    if is_best:
                        best_psnr = vp
                    best_note = " best" if is_best else ""
                    phase_note = " prior_warmup" if in_prior_warmup else ""
                    log_message(
                        f"[stage2][val]{phase_note} psnr={vp:.3f} "
                        f"ssim={vs:.4f}{best_note}",
                        log_file,
                    )
                    if args.val_direct:
                        direct_p, direct_s = validate(
                            unwrap_model(model),
                            val_loader,
                            device,
                            prior_mode="direct",
                            val_seed=args.val_seed,
                        )
                        log_message(
                            f"[stage2][val][direct] psnr={direct_p:.3f} "
                            f"ssim={direct_s:.4f}",
                            log_file,
                        )
                    if args.val_teacher:
                        teacher_p, teacher_s = validate_teacher(
                            unwrap_model(model),
                            val_loader,
                            device,
                        )
                        log_message(
                            f"[stage2][val][teacher] psnr={teacher_p:.3f} "
                            f"ssim={teacher_s:.4f}",
                            log_file,
                        )
                    if is_best:
                        state_dict = stage2_state_dict(unwrap_model(model))
                        save_checkpoint(
                            {
                                "training_version": 10,
                                "training_recipe": (
                                    "paper_stage2" if args.paper_stage2 else "custom"
                                ),
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
                if args.prior_mode == "direct":
                    unwrap_model(model).lcdm.eval()
                elif args.lcdm_lr_scale <= 0:
                    unwrap_model(model).lcdm.eval()
                if args.backbone_lr_scale <= 0:
                    unwrap_model(model).backbone.eval()
                if args.prior_lr_scale <= 0:
                    unwrap_model(model).sfpg_star.eval()
                if args.refiner_lr_scale <= 0:
                    unwrap_model(model).output_refiner.eval()

            if (it + 1) % args.ckpt_every == 0 or (it + 1) == args.total_iters:
                if is_main_process():
                    state_dict = stage2_state_dict(unwrap_model(model))
                    save_checkpoint(
                        {
                            "training_version": 10,
                            "training_recipe": (
                                "paper_stage2" if args.paper_stage2 else "custom"
                            ),
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
