import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .datasets import PairedUIRDataset
    from .metrics import (
        ExternalMetric,
        LPIPSMetric,
        URankerMetric,
        psnr,
        ssim,
        uciqe,
        uif,
        uiqm,
    )
    from .models import UniUIR, UniUIRConfig
except ImportError:
    from datasets import PairedUIRDataset
    from metrics import (
        ExternalMetric,
        LPIPSMetric,
        URankerMetric,
        psnr,
        ssim,
        uciqe,
        uif,
        uiqm,
    )
    from models import UniUIR, UniUIRConfig


REFERENCE_METRICS = {"psnr", "ssim", "lpips"}
NO_REFERENCE_METRICS = {"uciqe", "uiqm", "uranker", "uiqi"}
PAIRWISE_INPUT_METRICS = {"uif"}
ALL_METRICS = REFERENCE_METRICS | NO_REFERENCE_METRICS | PAIRWISE_INPUT_METRICS
DEFAULT_METRICS = ["psnr", "ssim", "lpips", "uciqe", "uiqm", "uranker", "uif"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["default"],
        help=(
            "default, all, or any of: "
            + ", ".join(sorted(ALL_METRICS))
        ),
    )
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--csv_out", type=str, default="metrics_per_image.csv")
    parser.add_argument("--json_out", type=str, default="metrics_summary.json")
    parser.add_argument("--uranker_ckpt", type=str, default=None)
    parser.add_argument(
        "--uiqi_callable",
        type=str,
        default=None,
        help=(
            "released UIQI callable as `module:function` or "
            "`/path/to/file.py:function`; it must accept RGB uint8 HWC"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    model_parser = subparsers.add_parser(
        "model",
        help="run UniUIR inference and evaluate the outputs",
    )
    model_parser.add_argument("--ckpt", required=True)
    model_parser.add_argument(
        "--data_root",
        required=True,
        help="paired dataset containing input/ and GT/",
    )
    model_parser.add_argument("--save_restored", type=str, default=None)
    model_parser.add_argument("--depth_backend", choices=["hf", "dummy"], default="hf")
    model_parser.add_argument(
        "--prior_mode",
        choices=["direct", "diffusion"],
        default=None,
        help="default: restore from checkpoint arguments",
    )
    model_parser.add_argument("--diffusion_anchor_strength", type=float, default=None)
    model_parser.add_argument(
        "--diffusion_stochastic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    model_parser.add_argument("--diffusion_state_clip", type=float, default=None)
    model_parser.add_argument("--diffusion_samples", type=int, default=None)
    model_parser.add_argument("--seed", type=int, default=2025)
    model_parser.add_argument("--num_workers", type=int, default=2)
    _add_metric_arguments(model_parser)

    folder_parser = subparsers.add_parser(
        "folders",
        help="evaluate an existing restored-image folder",
    )
    folder_parser.add_argument("--restored_dir", required=True)
    folder_parser.add_argument("--input_dir", default=None)
    folder_parser.add_argument("--gt_dir", default=None)
    _add_metric_arguments(folder_parser)
    return parser.parse_args()


def _resolve_metrics(requested: Iterable[str]) -> List[str]:
    requested = [value.lower() for value in requested]
    if requested == ["default"]:
        return list(DEFAULT_METRICS)
    if requested == ["all"]:
        return sorted(ALL_METRICS)
    unknown = sorted(set(requested) - ALL_METRICS)
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}")
    return list(dict.fromkeys(requested))


def _pil_to_tensor(image: Image.Image, resize: int) -> torch.Tensor:
    image = image.convert("RGB")
    if resize > 0:
        image = image.resize((resize, resize), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _save_tensor(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_tensor_to_uint8(image)).save(path)


class MetricRunner:
    def __init__(
        self,
        metric_names: List[str],
        device: torch.device,
        uranker_checkpoint: Optional[str],
        uiqi_callable: Optional[str],
    ):
        self.metric_names = metric_names
        self.lpips = (
            LPIPSMetric(device=device)
            if "lpips" in metric_names
            else None
        )
        if "uranker" in metric_names:
            if not uranker_checkpoint:
                raise RuntimeError(
                    "URanker was requested; pass --uranker_ckpt with the official "
                    "URanker_ckpt.pth."
                )
            self.uranker = URankerMetric(uranker_checkpoint, device)
        else:
            self.uranker = None
        if "uiqi" in metric_names:
            if not uiqi_callable:
                raise RuntimeError(
                    "UIQI was requested, but its trained regressor is not public in "
                    "this project. Pass --uiqi_callable module:function after "
                    "obtaining the authors' implementation."
                )
            self.uiqi = ExternalMetric(uiqi_callable)
        else:
            self.uiqi = None

    @torch.no_grad()
    def evaluate(
        self,
        restored: torch.Tensor,
        raw: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        if restored.ndim == 3:
            restored = restored.unsqueeze(0)
        if raw is not None and raw.ndim == 3:
            raw = raw.unsqueeze(0)
        if target is not None and target.ndim == 3:
            target = target.unsqueeze(0)
        if REFERENCE_METRICS.intersection(self.metric_names) and target is None:
            raise RuntimeError("PSNR/SSIM/LPIPS require matching GT images")
        if "uif" in self.metric_names and raw is None:
            raise RuntimeError("UIF requires matching raw input images")

        scores: Dict[str, float] = {}
        if "psnr" in self.metric_names:
            scores["psnr"] = psnr(restored, target)[0]
        if "ssim" in self.metric_names:
            scores["ssim"] = ssim(restored, target)[0]
        if self.lpips is not None:
            scores["lpips"] = self.lpips(restored, target)[0]
        restored_single = restored[0]
        restored_uint8 = _tensor_to_uint8(restored_single)
        if "uciqe" in self.metric_names:
            scores["uciqe"] = uciqe(restored_uint8)
        if "uiqm" in self.metric_names:
            scores["uiqm"] = uiqm(restored_uint8)
        if self.uranker is not None:
            scores["uranker"] = self.uranker(restored)[0]
        if "uif" in self.metric_names:
            scores["uif"] = uif(restored_uint8, _tensor_to_uint8(raw[0]))
        if self.uiqi is not None:
            scores["uiqi"] = self.uiqi(restored_uint8)
        return scores


def _load_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[UniUIR, dict]:
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    state = checkpoint["model"] if "model" in checkpoint else checkpoint

    state_clip = args.diffusion_state_clip
    if state_clip is None:
        state_clip = checkpoint_args.get("diffusion_state_clip", 0.0)
    latent_scale = checkpoint_args.get("latent_scale", 64.0)
    config = UniUIRConfig(
        depth_backend=args.depth_backend,
        latent_scale=latent_scale,
        diffusion_state_clip=state_clip,
    )
    model = UniUIR(config).to(device)
    required_prefixes = ("backbone.", "sfpg_star.", "lcdm.")
    absent = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in state)
    ]
    if absent:
        raise RuntimeError(f"checkpoint is missing Stage-II modules: {absent}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    relevant_missing = [
        key for key in missing if key.startswith(required_prefixes)
    ]
    if relevant_missing or unexpected:
        raise RuntimeError(
            "checkpoint load mismatch: "
            f"missing={relevant_missing}, unexpected={unexpected}"
        )
    model.eval()
    model.depth_predictor.eval()
    return model, checkpoint_args


def _checkpoint_option(
    cli_value,
    checkpoint_args: dict,
    name: str,
    default,
):
    return checkpoint_args.get(name, default) if cli_value is None else cli_value


@torch.no_grad()
def evaluate_model(args: argparse.Namespace, metrics: List[str]) -> List[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_args = _load_model(args, device)
    prior_mode = _checkpoint_option(
        args.prior_mode,
        checkpoint_args,
        "prior_mode",
        "diffusion",
    )
    anchor = _checkpoint_option(
        args.diffusion_anchor_strength,
        checkpoint_args,
        "diffusion_anchor_strength",
        1.0,
    )
    stochastic = _checkpoint_option(
        args.diffusion_stochastic,
        checkpoint_args,
        "diffusion_stochastic",
        True,
    )
    samples = _checkpoint_option(
        args.diffusion_samples,
        checkpoint_args,
        "diffusion_eval_samples",
        1,
    )
    if samples < 1:
        raise ValueError("--diffusion_samples must be positive")

    dataset = PairedUIRDataset(
        args.data_root,
        augment=False,
        resize_to=args.resize if args.resize > 0 else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    runner = MetricRunner(
        metrics,
        device,
        args.uranker_ckpt,
        args.uiqi_callable,
    )
    save_root = Path(args.save_restored) if args.save_restored else None
    rows: List[dict] = []
    print(
        "[evaluate] "
        f"resize={args.resize} prior_mode={prior_mode} anchor={anchor} "
        f"stochastic={stochastic} samples={samples} seed={args.seed}"
    )
    for index, batch in enumerate(loader):
        raw = batch["x_lq"].to(device)
        target = batch["x_gt"].to(device)
        generator = None
        if prior_mode == "diffusion":
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + index)
        output = model(
            raw,
            mode="inference",
            prior_mode=prior_mode,
            sample_generator=generator,
            diffusion_anchor_strength=anchor,
            diffusion_stochastic=stochastic,
            diffusion_samples=samples,
        )
        restored = output["x_hq"].clamp(0, 1)
        name = batch["name"][0]
        row = {"name": name}
        row.update(runner.evaluate(restored, raw, target))
        rows.append(row)
        if save_root is not None:
            _save_tensor(restored[0], save_root / name)
        print(
            f"[{index + 1:>3d}/{len(loader)}] {name} "
            + " ".join(f"{key}={value:.6f}" for key, value in row.items() if key != "name")
        )
    return rows


def _list_images(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"image directory not found: {folder}")
    return {
        path.name: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def evaluate_folders(args: argparse.Namespace, metrics: List[str]) -> List[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restored_files = _list_images(Path(args.restored_dir))
    raw_files = _list_images(Path(args.input_dir)) if args.input_dir else {}
    target_files = _list_images(Path(args.gt_dir)) if args.gt_dir else {}
    if not restored_files:
        raise RuntimeError("no restored images found")
    runner = MetricRunner(
        metrics,
        device,
        args.uranker_ckpt,
        args.uiqi_callable,
    )
    rows: List[dict] = []
    for index, (name, restored_path) in enumerate(restored_files.items()):
        raw_path = raw_files.get(name)
        target_path = target_files.get(name)
        if PAIRWISE_INPUT_METRICS.intersection(metrics) and raw_path is None:
            raise FileNotFoundError(f"missing raw input for {name}")
        if REFERENCE_METRICS.intersection(metrics) and target_path is None:
            raise FileNotFoundError(f"missing GT for {name}")
        restored = _pil_to_tensor(Image.open(restored_path), args.resize).to(device)
        raw = (
            _pil_to_tensor(Image.open(raw_path), args.resize).to(device)
            if raw_path
            else None
        )
        target = (
            _pil_to_tensor(Image.open(target_path), args.resize).to(device)
            if target_path
            else None
        )
        row = {"name": name}
        row.update(runner.evaluate(restored, raw, target))
        rows.append(row)
        print(
            f"[{index + 1:>3d}/{len(restored_files)}] {name} "
            + " ".join(f"{key}={value:.6f}" for key, value in row.items() if key != "name")
        )
    return rows


def _write_results(
    rows: List[dict],
    metrics: List[str],
    csv_path: str,
    json_path: str,
    args: argparse.Namespace,
) -> dict:
    if not rows:
        raise RuntimeError("evaluation produced no rows")
    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    with csv_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name"] + metrics)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "count": len(rows),
        "metrics": {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in metrics
        },
        "protocol": {
            "resize": args.resize,
            "rgb_range": "[0,1]",
            "model_outputs_quantized_for_reference_metrics": False,
            "uif_reference": "raw underwater input",
        },
        "arguments": vars(args),
    }
    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n=== averages ===")
    for metric, value in summary["metrics"].items():
        print(f"{metric:<9} {value:.6f}")
    print(f"per-image CSV: {csv_target}")
    print(f"summary JSON:  {json_target}")
    return summary


def main() -> None:
    args = parse_args()
    metrics = _resolve_metrics(args.metrics)
    if args.mode == "model":
        rows = evaluate_model(args, metrics)
    else:
        rows = evaluate_folders(args, metrics)
    _write_results(rows, metrics, args.csv_out, args.json_out, args)


if __name__ == "__main__":
    main()
