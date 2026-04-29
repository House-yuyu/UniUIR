import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (standard DDPM form).
# ---------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Token-level denoising block: Self-Attn(Z_t) + Cross-Attn(Z_t <- C) + FFN, time-modulated.
# ---------------------------------------------------------------------------
class DenoiseBlock(nn.Module):
    def __init__(self, dim, n_heads=4, ffn_mult=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.t_proj = nn.Linear(dim, dim * 2)

    def forward(self, x, cond, t_emb):
        # x: (B, 1, dim);  cond: (B, N, dim);  t_emb: (B, dim)
        scale, shift = self.t_proj(F.silu(t_emb)).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        h = self.norm1(x) * (1 + scale) + shift
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h

        h = self.norm2(x)
        h, _ = self.cross_attn(h, cond, cond, need_weights=False)
        x = x + h

        x = x + self.ffn(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Token-level "UNet" denoising network: predicts epsilon given (Z_t, C, t).
# ---------------------------------------------------------------------------
class DenoisingNetwork(nn.Module):
    def __init__(self, prior_dim=256, n_blocks=4, n_heads=4):
        super().__init__()
        self.time_embed = TimeEmbedding(prior_dim)
        self.in_proj = nn.Linear(prior_dim, prior_dim)
        self.blocks = nn.ModuleList([
            DenoiseBlock(prior_dim, n_heads=n_heads) for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(prior_dim, prior_dim)

    def forward(self, z_t, cond, t):
        """
        z_t  : (B, prior_dim)
        cond : (B, N, prior_dim)  -- output of SFPG*
        t    : (B,)               -- discrete timesteps
        returns predicted noise eps (B, prior_dim).
        """
        t_emb = self.time_embed(t)
        x = self.in_proj(z_t).unsqueeze(1)  # (B, 1, dim)
        for blk in self.blocks:
            x = blk(x, cond, t_emb)
        return self.out_proj(x.squeeze(1))


# ---------------------------------------------------------------------------
# LCDM wrapper: forward q(Z_t | Z) and reverse sampling.
# ---------------------------------------------------------------------------
class LCDM(nn.Module):
    def __init__(self,
                 prior_dim=256,
                 num_steps=4,
                 alpha_start=0.99,
                 alpha_end=0.1,
                 n_blocks=4,
                 n_heads=4):
        super().__init__()
        self.T = num_steps
        self.prior_dim = prior_dim
        # Linear schedule on alpha_t (paper sets a_1 = 0.99, a_T = 0.1).
        alphas = torch.linspace(alpha_start, alpha_end, num_steps)
        betas = 1.0 - alphas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)            # (T,)
        self.register_buffer("betas", betas)              # (T,)
        self.register_buffer("alphas_bar", alphas_bar)    # (T,)

        self.denoiser = DenoisingNetwork(prior_dim=prior_dim,
                                         n_blocks=n_blocks, n_heads=n_heads)

    # -- forward (training) ---------------------------------------------------
    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        a_bar = self.alphas_bar[t].view(-1, 1)
        return torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise, noise

    def training_step(self, z0, cond):
        """
        Returns predicted Z_hat reconstructed from a single sampled t (used to
        compute L_diff = L1(Z, Z_hat) per Eq. 4) along with the predicted noise
        and ground-truth noise (handy for an auxiliary noise-prediction loss).
        """
        B = z0.size(0)
        t = torch.randint(0, self.T, (B,), device=z0.device)
        z_t, noise = self.q_sample(z0, t)
        eps_pred = self.denoiser(z_t, cond, t)

        a_bar = self.alphas_bar[t].view(-1, 1)
        # Reconstruct Z_0 from (Z_t, eps_pred): standard DDPM formula.
        z0_pred = (z_t - torch.sqrt(1.0 - a_bar) * eps_pred) / torch.sqrt(a_bar)
        return z0_pred, eps_pred, noise

    # -- reverse (inference) --------------------------------------------------
    @torch.no_grad()
    def sample(self, cond, shape=None):
        """Iterative DDPM-style denoising for T steps -> Z_hat."""
        B = cond.size(0)
        if shape is None:
            shape = (B, self.prior_dim)
        z = torch.randn(shape, device=cond.device)
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=cond.device, dtype=torch.long)
            eps = self.denoiser(z, cond, t)
            a_t = self.alphas[i]
            a_bar = self.alphas_bar[i]
            coeff = (1.0 - a_t) / torch.sqrt(1.0 - a_bar)
            mean = (z - coeff * eps) / torch.sqrt(a_t)
            if i > 0:
                noise = torch.randn_like(z)
                z = mean + torch.sqrt(1.0 - a_t) * noise
            else:
                z = mean
        return z


if __name__ == "__main__":
    lcdm = LCDM(prior_dim=256, num_steps=4).cuda()
    z0 = torch.randn(2, 256).cuda()
    c = torch.randn(2, 5, 256).cuda()
    z0_pred, eps_pred, eps = lcdm.training_step(z0, c)
    print("z0_pred:", z0_pred.shape, "eps_pred:", eps_pred.shape)
    z_hat = lcdm.sample(c)
    print("z_hat:", z_hat.shape)
