from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .mmoe_uir import MMoEUIR
from .sfpg import SFPG, SFPGStar
from .lcdm import LCDM
from .depth_extractor import build_depth_predictor


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
                         n_blocks=cfg.lcdm_blocks,
                         n_heads=cfg.lcdm_heads)
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
        self.sfpg_star.eval()
        self.lcdm.eval()

    # ------------------------------------------------------------------
    def forward(self, x_lq, x_gt=None, mode="stage1"):
        """
        Args
        ----
        x_lq : (B, 3, H, W)         low-quality input.
        x_gt : (B, 3, H, W) or None ground-truth (required for stage1/stage2).
        mode : 'stage1' | 'stage2' | 'inference'

        Returns dict with keys depending on mode:
          stage1   : x_hq, depth (D_lq), depth_hq (D_hq), Z
          stage2   : x_hq, depth, Z, Z_hat, eps_pred, eps
          inference: x_hq, depth, Z_hat
        """
        d_lq = self.get_depth(x_lq)

        if mode == "stage1":
            assert x_gt is not None
            Z = self.sfpg(x_lq, x_gt)                              # (B, C_hat)
            x_hq = self.backbone(x_lq, prior=Z, depth=d_lq)
            d_hq = self.depth_predictor(x_hq.clamp(0, 1))
            return {"x_hq": x_hq, "depth": d_lq, "depth_hq": d_hq, "Z": Z}

        if mode == "stage2":
            assert x_gt is not None
            with torch.no_grad():
                Z = self.sfpg(x_lq, x_gt)
            C = self.sfpg_star(x_lq)
            Z_hat, eps_pred, eps = self.lcdm.training_step(Z, C)
            x_hq = self.backbone(x_lq, prior=Z_hat, depth=d_lq)
            return {"x_hq": x_hq, "depth": d_lq, "Z": Z,
                    "Z_hat": Z_hat, "eps_pred": eps_pred, "eps": eps}

        if mode == "inference":
            C = self.sfpg_star(x_lq)
            Z_hat = self.lcdm.sample(C)
            x_hq = self.backbone(x_lq, prior=Z_hat, depth=d_lq)
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
