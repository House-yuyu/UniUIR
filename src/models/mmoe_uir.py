import math
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.layers import DropPath

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    selective_scan_fn = None


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert HAS_MAMBA, "mamba_ssm is required. pip install mamba-ssm causal-conv1d"
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))
        del x_proj

        dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max,
                         dt_init_floor, **factory_kwargs)
            for _ in range(4)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [x.view(B, -1, L),
             torch.transpose(x, 2, 3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, 1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Water Mixture-of-Experts (W-MoE)
# ---------------------------------------------------------------------------
class StripedConv2d(nn.Module):
    def __init__(self, in_ch, kernel_size, depthwise=False):
        super().__init__()
        p = kernel_size // 2
        g = in_ch if depthwise else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, (1, kernel_size), padding=(0, p), groups=g),
            nn.Conv2d(in_ch, in_ch, (kernel_size, 1), padding=(p, 0), groups=g),
        )

    def forward(self, x):
        return self.conv(x)


def channel_shuffle(x, groups=2):
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, -1, h, w)


class Expert(nn.Module):
    """Low-rank expert E_i: T^3 ( T^1(F_a) odot T^2(F_b) )."""
    def __init__(self, in_ch, low_dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, 1)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, 1)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, 1)

    def forward(self, x, k):
        return self.conv_3(self.conv_2(k) * self.conv_1(x))


class Router(nn.Module):
    def __init__(self, in_ch, num_experts):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x):
        return self.body(x)


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, x, k):
        logits = self.gate(x)
        weights = F.softmax(logits, dim=1, dtype=torch.float).to(x.dtype)
        topk_weights, topk_idx = torch.topk(weights, self.num_expert, dim=1)
        out = x.clone()
        if self.training:
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topk_idx, weights.gather(1, topk_idx))
            for i, expert in enumerate(self.experts):
                out = out + expert(x, k) * mask[:, i:i + 1, None, None]
        else:
            # Per-sample routing at inference; simple loop suffices for batch=1.
            for b in range(x.size(0)):
                for j in range(self.num_expert):
                    e = topk_idx[b, j].item()
                    w = topk_weights[b, j]
                    out[b:b + 1] = out[b:b + 1] + self.experts[e](x[b:b + 1], k[b:b + 1]) * w
        return out


class WMoE(nn.Module):
    def __init__(self, in_ch, num_experts=3, topk=2, use_shuffle=False, lr_space="exp"):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, 1),
        )
        self.conv_strip = nn.Sequential(StripedConv2d(in_ch, 3, depthwise=True), nn.GELU())

        if lr_space == "linear":
            grow = lambda i: i + 2
        elif lr_space == "exp":
            grow = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(lr_space)

        self.moe_layer = MoELayer(
            experts=[Expert(in_ch, grow(i)) for i in range(num_experts)],
            gate=Router(in_ch, num_experts),
            num_expert=topk,
        )
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        x = self.conv_1(x)
        if self.use_shuffle:
            x = channel_shuffle(x, 2)
        x, k = torch.chunk(x, 2, dim=1)
        x = self.conv_strip(x)
        x = self.moe_layer(x, k)
        return self.proj(x)


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MoEFFN(nn.Module):
    def __init__(self, in_ch, num_experts=3, topk=2, prior_dim=256):
        super().__init__()
        self.norm_1 = LayerNorm2d(in_ch, data_format='channels_first')
        self.block = WMoE(in_ch, num_experts=num_experts, topk=topk)
        # Prior modulation: project flat prior Z (B, prior_dim) to (B, in_ch, 1, 1).
        self.ln1 = nn.Linear(prior_dim, in_ch)
        self.ln2 = nn.Linear(prior_dim, in_ch)
        # Depth modulation: 3x3 conv on single-channel depth map -> in_ch attention.
        self.conv_depth = nn.Sequential(
            nn.Conv2d(1, in_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Start from identity prior modulation: scale ~= 1, bias ~= 0.
        nn.init.zeros_(self.ln1.weight)
        nn.init.zeros_(self.ln1.bias)
        nn.init.zeros_(self.ln2.weight)
        nn.init.zeros_(self.ln2.bias)

    def forward(self, x, prior=None, depth=None):
        # x: (B, C, H, W); depth: (B, 1, H, W) downsampled to current scale; prior: (B, prior_dim).
        res = x
        x = self.norm_1(x)
        if depth is not None:
            x = x * F.softmax(self.conv_depth(depth), dim=1)
        if prior is not None:
            # Keep prior modulation bounded at initialization to avoid exploding
            # activations across many stacked MMoE blocks.
            k1 = (1.0 + torch.tanh(self.ln1(prior))).unsqueeze(-1).unsqueeze(-1)
            k2 = torch.tanh(self.ln2(prior)).unsqueeze(-1).unsqueeze(-1)
            x = x * k1 + k2
        return self.block(x) + res


class MMoEB(nn.Module):
    def __init__(self, hidden_dim, drop_path=0., d_state=16, expand=2.,
                 num_experts=3, topk=2, prior_dim=256, attn_drop_rate=0.):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,
                                   expand=expand, dropout=attn_drop_rate)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.full((hidden_dim,), 1e-3))
        self.skip_scale2 = nn.Parameter(torch.full((hidden_dim,), 1e-3))
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.moe = MoEFFN(hidden_dim, num_experts=num_experts, topk=topk,
                          prior_dim=prior_dim)
        # Linear projections for prior modulation in Eq. (5).
        self.ln1 = nn.Linear(prior_dim, hidden_dim)
        self.ln2 = nn.Linear(prior_dim, hidden_dim)
        # Start from identity prior modulation: scale ~= 1, bias ~= 0.
        nn.init.zeros_(self.ln1.weight)
        nn.init.zeros_(self.ln1.bias)
        nn.init.zeros_(self.ln2.weight)
        nn.init.zeros_(self.ln2.bias)

    def forward(self, x_seq, x_size, prior=None, depth=None):
        # x_seq: (B, HW, C) -> (B, H, W, C)
        B, L, C = x_seq.shape
        H, W = x_size
        x = x_seq.view(B, H, W, C).contiguous()
        residual = x

        # VSSM branch (Eq. 5)
        z = self.ln_1(x)
        if prior is not None:
            # Bound the scale/bias injected from the degradation prior so the
            # pre-trained/randomly initialized prior projection does not blow up
            # the SS2D branch on the very first forward pass.
            k1 = (1.0 + torch.tanh(self.ln1(prior))).view(B, 1, 1, C)
            k2 = torch.tanh(self.ln2(prior)).view(B, 1, 1, C)
            z = z * k1 + k2
        x = residual + self.skip_scale.view(1, 1, 1, C) * self.drop_path(self.self_attention(z))

        # W-MoE branch (Eq. 6,7) -- operates in channels_first.
        x_cf = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x_cf = self.moe(x_cf, prior=prior, depth=depth)
        x = x + self.skip_scale2.view(1, 1, 1, C) * x_cf.permute(0, 2, 3, 1).contiguous()

        return x.view(B, H * W, C).contiguous()


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "b c h w -> b (h w) c").contiguous()


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        return rearrange(x, "b c h w -> b (h w) c").contiguous()


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        return rearrange(x, "b c h w -> b (h w) c").contiguous()


# ---------------------------------------------------------------------------
# MMoE-UIR: 4-stage U-shape backbone with prior + depth injection
# ---------------------------------------------------------------------------
class MMoEUIR(nn.Module):
    """Paper default: dim=32, num_blocks=[3, 5, 6, 6], num_experts=3, topk=2."""
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=(3, 5, 6, 6),
                 num_refinement_blocks=2,
                 mlp_ratio=2.,
                 prior_dim=256,
                 num_experts=3,
                 topk=2,
                 drop_path_rate=0.,
                 bias=False):
        super().__init__()
        self.prior_dim = prior_dim
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        base_d_state = 4

        def make_stage(n_blocks, d_in, d_state):
            return nn.ModuleList([
                MMoEB(hidden_dim=d_in, drop_path=drop_path_rate, d_state=d_state,
                      expand=mlp_ratio, num_experts=num_experts, topk=topk,
                      prior_dim=prior_dim)
                for _ in range(n_blocks)
            ])

        self.encoder_level1 = make_stage(num_blocks[0], dim, base_d_state)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = make_stage(num_blocks[1], dim * 2, base_d_state * 2)
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = make_stage(num_blocks[2], dim * 4, base_d_state * 4)
        self.down3_4 = Downsample(dim * 4)
        self.latent = make_stage(num_blocks[3], dim * 8, base_d_state * 8)

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = make_stage(num_blocks[2], dim * 4, base_d_state * 4)

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = make_stage(num_blocks[1], dim * 2, base_d_state * 2)

        self.up2_1 = Upsample(dim * 2)
        # No 1x1 reduction at level-1 (matches Restormer/MMoEUIR convention).
        self.decoder_level1 = make_stage(num_blocks[0], dim * 2, base_d_state * 2)
        self.refinement = make_stage(num_refinement_blocks, dim * 2, base_d_state * 2)

        self.output = nn.Conv2d(dim * 2, out_channels, 3, padding=1, bias=bias)
        # Start from an identity-like restoration mapping: x_hq ~= x_lq.
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    @staticmethod
    def _down_depth(d, scale):
        if d is None:
            return None
        return F.avg_pool2d(d, scale) if scale > 1 else d

    def _run_stage(self, layers, x, size, prior, depth):
        for layer in layers:
            x = layer(x, size, prior=prior, depth=depth)
        return x

    def forward(self, inp_img, prior=None, depth=None):
        """
        inp_img: (B, 3, H, W) low-quality underwater image X_lq.
        prior:   (B, prior_dim) compact degradation prior Z.
        depth:   (B, 1, H, W)   D_lq from Depth Anything V2 (already resized to H, W).
        """
        _, _, H, W = inp_img.shape
        d1 = self._down_depth(depth, 1)
        d2 = self._down_depth(depth, 2)
        d3 = self._down_depth(depth, 4)
        d4 = self._down_depth(depth, 8)

        x1 = self.patch_embed(inp_img)
        x1 = self._run_stage(self.encoder_level1, x1, (H, W), prior, d1)

        x2 = self.down1_2(x1, H, W)
        x2 = self._run_stage(self.encoder_level2, x2, (H // 2, W // 2), prior, d2)

        x3 = self.down2_3(x2, H // 2, W // 2)
        x3 = self._run_stage(self.encoder_level3, x3, (H // 4, W // 4), prior, d3)

        x4 = self.down3_4(x3, H // 4, W // 4)
        latent = self._run_stage(self.latent, x4, (H // 8, W // 8), prior, d4)

        d3_up = self.up4_3(latent, H // 8, W // 8)
        d3_up = torch.cat([d3_up, x3], dim=2)
        d3_up = rearrange(d3_up, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        d3_up = self.reduce_chan_level3(d3_up)
        d3_up = rearrange(d3_up, "b c h w -> b (h w) c").contiguous()
        d3_up = self._run_stage(self.decoder_level3, d3_up, (H // 4, W // 4), prior, d3)

        d2_up = self.up3_2(d3_up, H // 4, W // 4)
        d2_up = torch.cat([d2_up, x2], dim=2)
        d2_up = rearrange(d2_up, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        d2_up = self.reduce_chan_level2(d2_up)
        d2_up = rearrange(d2_up, "b c h w -> b (h w) c").contiguous()
        d2_up = self._run_stage(self.decoder_level2, d2_up, (H // 2, W // 2), prior, d2)

        d1_up = self.up2_1(d2_up, H // 2, W // 2)
        d1_up = torch.cat([d1_up, x1], dim=2)
        d1_up = self._run_stage(self.decoder_level1, d1_up, (H, W), prior, d1)
        d1_up = self._run_stage(self.refinement, d1_up, (H, W), prior, d1)

        out = rearrange(d1_up, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        return self.output(out) + inp_img


if __name__ == "__main__":
    model = MMoEUIR(dim=32, num_blocks=(3, 5, 6, 6), num_refinement_blocks=2).cuda()
    x = torch.randn(1, 3, 128, 128).cuda()
    z = torch.randn(1, 256).cuda()
    d = torch.randn(1, 1, 128, 128).cuda()
    y = model(x, prior=z, depth=d)
    print("output:", y.shape)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
