import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


# ---------------------------------------------------------------------------
class SFPGBase(nn.Module):
    """Shared backbone of SFPG / SFPG*.

    Args
    ----
    in_ch          : number of input channels (6 for SFPG, 3 for SFPG*).
    use_unshuffle  : True for SFPG (downsample by PixelUnshuffle), False for SFPG*.
    inner_ch       : intermediate channel width.
    prior_dim      : C_hat -- final prior channel dimension (256 in paper).
    num_prompts    : N    -- number of task-related prompt slots.
    """
    def __init__(self,
                 in_ch=6,
                 use_unshuffle=True,
                 inner_ch=64,
                 prior_dim=256,
                 num_prompts=5,
                 num_resblocks=3):
        super().__init__()
        self.num_prompts = num_prompts
        self.prior_dim = prior_dim

        if use_unshuffle:
            # PixelUnshuffle by 4 -> 16x channels.
            self.head = nn.Sequential(
                nn.PixelUnshuffle(4),
                nn.Conv2d(in_ch * 16, inner_ch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, inner_ch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.amp_branch = nn.Sequential(*[ResBlock(inner_ch) for _ in range(num_resblocks)])
        self.phase_branch = nn.Sequential(*[ResBlock(inner_ch) for _ in range(num_resblocks)])

        self.tail = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inner_ch, inner_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inner_ch, num_prompts, 1),
        )

        # Task-related prompts P in R^{N x C_hat}.
        self.prompts = nn.Parameter(torch.randn(num_prompts, prior_dim) * 0.02)

    def _spec_branch(self, x):
        # 2D rFFT -> separate amplitude/phase ResBlocks -> iFFT back to spatial.
        f = torch.fft.rfft2(x, norm="ortho")
        amp = torch.abs(f)
        pha = torch.angle(f)
        amp = self.amp_branch(amp)
        pha = self.phase_branch(pha)
        f_new = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))
        return torch.fft.irfft2(f_new, s=x.shape[-2:], norm="ortho")

    def forward(self, x):
        """
        x : (B, in_ch, H, W)
        returns:
            S  : (B, N)            softmax weights over prompts.
            P  : (N, C_hat)        the learnable prompt bank (shared).
        """
        h = self.head(x)
        h = self._spec_branch(h)
        h = self.tail(h)                            # (B, N, h, w)
        S = F.adaptive_avg_pool2d(h, 1).flatten(1)  # (B, N)
        S = F.softmax(S, dim=1)
        return S, self.prompts


# ---------------------------------------------------------------------------
class SFPG(nn.Module):
    """Stage-I prior extractor: takes concat[X_lq, X_gt] -> Z (B, C_hat)."""
    def __init__(self, prior_dim=256, num_prompts=5, inner_ch=64):
        super().__init__()
        self.net = SFPGBase(in_ch=6, use_unshuffle=True,
                            inner_ch=inner_ch, prior_dim=prior_dim,
                            num_prompts=num_prompts)

    def forward(self, x_lq, x_gt):
        x = torch.cat([x_lq, x_gt], dim=1)
        S, P = self.net(x)
        # Eq. (12): Z = P^T @ S  -> (B, C_hat)
        Z = S @ P
        return Z


# ---------------------------------------------------------------------------
class SFPGStar(nn.Module):
    """Inference-time / diffusion-condition encoder: takes X_lq only.

    Returns the per-prompt conditional embedding C in R^{B x N x C_hat}, used as the
    cross-condition for the LCDM denoising UNet (Sec. III-D).
    """
    def __init__(self, prior_dim=256, num_prompts=5, inner_ch=64):
        super().__init__()
        self.net = SFPGBase(in_ch=3, use_unshuffle=False,
                            inner_ch=inner_ch, prior_dim=prior_dim,
                            num_prompts=num_prompts)
        self.num_prompts = num_prompts
        self.prior_dim = prior_dim

    def forward(self, x_lq):
        S, P = self.net(x_lq)            # S: (B,N), P: (N,C_hat)
        # Spread into per-prompt tokens: C[b, n, :] = S[b, n] * P[n, :]
        C = S.unsqueeze(-1) * P.unsqueeze(0)  # (B, N, C_hat)
        return C


if __name__ == "__main__":
    sfpg = SFPG().cuda()
    x_lq = torch.randn(2, 3, 128, 128).cuda()
    x_gt = torch.randn(2, 3, 128, 128).cuda()
    Z = sfpg(x_lq, x_gt)
    print("Z:", Z.shape)  # (2, 256)

    sfpg_s = SFPGStar().cuda()
    C = sfpg_s(x_lq)
    print("C:", C.shape)  # (2, 5, 256)
