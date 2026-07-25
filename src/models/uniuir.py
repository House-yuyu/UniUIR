from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .mmoe_uir import MMoEUIR
from .sfpg import SFPG, SFPGStar
from .lcdm import LCDM
from .depth_extractor import build_depth_predictor


class OutputRefiner(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_lq, restored):
        correction = self.net(torch.cat([x_lq, restored], dim=1))
        return restored + correction


@dataclass
class UniUIRConfig:
    # MMoE-UIR
    dim: int = 32
    num_blocks: tuple = (3, 5, 6, 6)
    num_refinement_blocks: int = 2
    num_experts: int = 3
    topk: int = 2
    mlp_ratio: float = 2.0
    drop_path_rate: float = 0.0
    # SFPG
    prior_dim: int = 256
    num_prompts: int = 5
    sfpg_inner_ch: int = 64
    # LCDM
    diffusion_steps: int = 4
    alpha_start: float = 0.99
    alpha_end: float = 0.1
    latent_scale: float = 64.0
    diffusion_state_clip: float = 0.0
    lcdm_blocks: int = 4
    lcdm_heads: int = 4
    # Depth
    depth_backend: str = "hf"   # "hf" or "dummy"


class UniUIR(nn.Module):
    def __init__(self, cfg: UniUIRConfig = UniUIRConfig()):
        super().__init__()
        self.cfg = cfg

        self.backbone = MMoEUIR(
            dim=cfg.dim,
            num_blocks=cfg.num_blocks,
            num_refinement_blocks=cfg.num_refinement_blocks,
            mlp_ratio=cfg.mlp_ratio,
            prior_dim=cfg.prior_dim,
            num_experts=cfg.num_experts,
            topk=cfg.topk,
            drop_path_rate=cfg.drop_path_rate,
        )
        self.sfpg = SFPG(prior_dim=cfg.prior_dim,
                         num_prompts=cfg.num_prompts,
                         inner_ch=cfg.sfpg_inner_ch)
        self.sfpg_star = SFPGStar(prior_dim=cfg.prior_dim,
                                  num_prompts=cfg.num_prompts,
                                  inner_ch=cfg.sfpg_inner_ch)
        self.lcdm = LCDM(prior_dim=cfg.prior_dim,
                         num_steps=cfg.diffusion_steps,
                         alpha_start=cfg.alpha_start,
                         alpha_end=cfg.alpha_end,
                         latent_scale=cfg.latent_scale,
                         state_clip=cfg.diffusion_state_clip,
                         n_blocks=cfg.lcdm_blocks,
                         n_heads=cfg.lcdm_heads)
        self.output_refiner = OutputRefiner()
        self.depth_predictor = build_depth_predictor(backend=cfg.depth_backend)

        # Depth predictor is frozen.
        for p in self.depth_predictor.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_depth(self, x_lq):
        return self.depth_predictor(x_lq)

    # ------------------------------------------------------------------
    def freeze_for_stage2(self):
        """Stage II: freeze SFPG (Z is the diffusion target, fixed)."""
        for p in self.sfpg.parameters():
            p.requires_grad_(False)
        self.sfpg.eval()

    # ------------------------------------------------------------------
    def freeze_for_stage1(self):
        """Stage I: freeze modules introduced only for Stage II/inference."""
        for p in self.sfpg_star.parameters():
            p.requires_grad_(False)
        for p in self.lcdm.parameters():
            p.requires_grad_(False)
        for p in self.output_refiner.parameters():
            p.requires_grad_(False)
        self.sfpg_star.eval()
        self.lcdm.eval()
        self.output_refiner.eval()

    # ------------------------------------------------------------------
    def forward(self, x_lq, x_gt=None, mode="stage1",
                prior_mode="direct", sample_noise=None,
                sample_generator=None, prior_lq=None, prior_gt=None,
                compute_eps_aux=True, diffusion_target="teacher",
                diffusion_train_start="terminal",
                diffusion_anchor_strength=1.0,
                diffusion_stochastic=True,
                diffusion_samples=1):

        prior_lq = x_lq if prior_lq is None else prior_lq
        prior_gt = x_gt if prior_gt is None else prior_gt
        if diffusion_samples < 1:
            raise ValueError(
                f"diffusion_samples must be at least 1, got {diffusion_samples}"
            )

        if mode == "prior_distill":
            C = self.sfpg_star(prior_lq)
            if diffusion_target == "direct":
                Z = C.sum(dim=1).detach()
            elif diffusion_target == "teacher":
                assert prior_gt is not None
                with torch.no_grad():
                    Z = self.sfpg(prior_lq, prior_gt)
            else:
                raise ValueError(
                    f"unknown diffusion_target: {diffusion_target}"
                )
            if prior_mode == "direct":
                Z_direct = C.sum(dim=1)
                Z_hat = Z_direct
                Z_diffusion = Z_hat
                eps_pred = eps = None
            elif prior_mode == "diffusion":
                Z_direct = C.sum(dim=1)
                Z_diffusion, eps_pred, eps = self.lcdm.training_step(
                    Z,
                    C,
                    compute_eps_aux=compute_eps_aux,
                    start_from_gaussian=(
                        diffusion_train_start == "gaussian"
                    ),
                    stochastic=diffusion_stochastic,
                )
                Z_hat = torch.lerp(
                    Z_direct,
                    Z_diffusion,
                    diffusion_anchor_strength,
                )
            else:
                raise ValueError(f"unknown prior_mode: {prior_mode}")
            return {"Z": Z, "Z_hat": Z_hat, "Z_direct": Z_direct,
                    "Z_diffusion": Z_diffusion,
                    "eps_pred": eps_pred, "eps": eps}

        d_lq = self.get_depth(x_lq)
        if mode == "stage1":
            assert x_gt is not None
            Z = self.sfpg(x_lq, x_gt)                              # (B, C_hat)
            x_hq = self.backbone(x_lq, prior=Z, depth=d_lq)
            d_hq = self.depth_predictor(x_hq.clamp(0, 1))
            return {"x_hq": x_hq, "depth": d_lq, "depth_hq": d_hq, "Z": Z}

        if mode == "stage2":
            assert x_gt is not None and prior_gt is not None
            C = self.sfpg_star(prior_lq)
            if diffusion_target == "direct":
                Z = C.sum(dim=1).detach()
            elif diffusion_target == "teacher":
                with torch.no_grad():
                    Z = self.sfpg(prior_lq, prior_gt)
            else:
                raise ValueError(
                    f"unknown diffusion_target: {diffusion_target}"
                )
            if prior_mode == "direct":
                Z_direct = C.sum(dim=1)
                Z_hat = Z_direct
                Z_diffusion = Z_hat
                eps_pred = eps = None
            elif prior_mode == "diffusion":
                Z_direct = C.sum(dim=1)
                Z_diffusion, eps_pred, eps = self.lcdm.training_step(
                    Z,
                    C,
                    compute_eps_aux=compute_eps_aux,
                    start_from_gaussian=(
                        diffusion_train_start == "gaussian"
                    ),
                    stochastic=diffusion_stochastic,
                )
                Z_hat = torch.lerp(
                    Z_direct,
                    Z_diffusion,
                    diffusion_anchor_strength,
                )
            else:
                raise ValueError(f"unknown prior_mode: {prior_mode}")
            x_hq = self.backbone(x_lq, prior=Z_hat, depth=d_lq)
            x_hq = self.output_refiner(x_lq, x_hq)
            return {"x_hq": x_hq, "depth": d_lq, "Z": Z,
                    "Z_hat": Z_hat, "Z_direct": Z_direct,
                    "Z_diffusion": Z_diffusion,
                    "eps_pred": eps_pred, "eps": eps}

        if mode == "teacher":
            assert x_gt is not None
            with torch.no_grad():
                Z = self.sfpg(x_lq, x_gt)
            x_hq = self.backbone(x_lq, prior=Z, depth=d_lq)
            x_hq = self.output_refiner(x_lq, x_hq)
            return {"x_hq": x_hq, "depth": d_lq, "Z": Z,
                    "Z_hat": Z, "eps_pred": None, "eps": None}

        if mode == "inference":
            if prior_mode == "self":

                Z_hat = self.sfpg(x_lq, x_lq)
            else:
                C = self.sfpg_star(x_lq)
            if prior_mode == "direct":
                Z_hat = C.sum(dim=1)
            elif prior_mode == "diffusion":
                diffusion_predictions = [
                    self.lcdm.sample(
                        C,
                        noise=sample_noise,
                        generator=sample_generator,
                        stochastic=diffusion_stochastic,
                    )
                    for _ in range(diffusion_samples)
                ]
                Z_diffusion = torch.stack(
                    diffusion_predictions,
                    dim=0,
                ).mean(dim=0)
                Z_hat = torch.lerp(
                    C.sum(dim=1),
                    Z_diffusion,
                    diffusion_anchor_strength,
                )
            elif prior_mode != "self":
                raise ValueError(f"unknown prior_mode: {prior_mode}")
            x_hq = self.backbone(x_lq, prior=Z_hat, depth=d_lq)
            x_hq = self.output_refiner(x_lq, x_hq)
            return {"x_hq": x_hq, "depth": d_lq, "Z_hat": Z_hat}

        raise ValueError(f"unknown mode: {mode}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = UniUIRConfig(depth_backend="dummy")
    model = UniUIR(cfg).cuda()
    x_lq = torch.rand(1, 3, 128, 128).cuda()
    x_gt = torch.rand(1, 3, 128, 128).cuda()
    out1 = model(x_lq, x_gt, mode="stage1")
    print({k: v.shape for k, v in out1.items()})
    out2 = model(x_lq, x_gt, mode="stage2")
    print({k: v.shape for k, v in out2.items()})
    out3 = model(x_lq, mode="inference")
    print({k: v.shape for k, v in out3.items()})
