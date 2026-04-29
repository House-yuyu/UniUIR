import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize(d: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalization to [0, 1]."""
    B = d.size(0)
    flat = d.view(B, -1)
    dmin = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    dmax = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    return (d - dmin) / (dmax - dmin + 1e-6)


# ---------------------------------------------------------------------------
class DepthAnythingV2HF(nn.Module):
    """Backend that uses the HuggingFace transformers checkpoint."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Large-hf"):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        mean = torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @torch.no_grad()
    def forward(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """rgb_01 : (B, 3, H, W) in [0, 1]; returns (B, 1, H, W) depth in [0, 1]."""
        H, W = rgb_01.shape[-2:]
        x = (rgb_01 - self.mean) / self.std
        # The HF wrapper expects pixel_values; the encoder needs sizes divisible by 14.
        h_pad = (14 - H % 14) % 14
        w_pad = (14 - W % 14) % 14
        if h_pad or w_pad:
            x = F.pad(x, (0, w_pad, 0, h_pad), mode="reflect")
        out = self.model(pixel_values=x).predicted_depth  # (B, h', w')
        if out.dim() == 3:
            out = out.unsqueeze(1)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return _normalize(out)


# ---------------------------------------------------------------------------
class DummyDepth(nn.Module):
    """Fallback predictor (frozen, returns the grayscale of the input)."""
    @torch.no_grad()
    def forward(self, rgb_01):
        return _normalize(rgb_01.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
def build_depth_predictor(backend="hf", **kwargs) -> nn.Module:
    """Factory.

    backend = "hf"    -> HuggingFace Depth-Anything-V2-Large-hf
    backend = "dummy" -> grayscale fallback (for environments without the weights)
    """
    if backend == "hf":
        try:
            return DepthAnythingV2HF(**kwargs)
        except Exception as e:
            print(f"[depth] HF backend unavailable ({e}); falling back to dummy.")
            return DummyDepth()
    elif backend == "dummy":
        return DummyDepth()
    raise ValueError(f"unknown depth backend: {backend}")


if __name__ == "__main__":
    net = build_depth_predictor(backend="dummy")
    x = torch.rand(2, 3, 128, 128)
    print(net(x).shape)
