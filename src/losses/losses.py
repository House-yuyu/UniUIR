import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_xy(x: torch.Tensor):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


class GradientLoss(nn.Module):
    """Eq. (3): | d_x diff | + | d_y diff | averaged over pixels."""
    def forward(self, pred, target):
        diff = pred - target
        dx, dy = gradient_xy(diff)
        return dx.abs().mean() + dy.abs().mean()


class EdgeAwareDepthLoss(nn.Module):
    """Eq. (2): L_depth = L1(D_pseudo, D_hq) + lambda_2 * L_grad."""
    def __init__(self, lambda_grad: float = 0.5):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.l1 = nn.L1Loss()
        self.grad = GradientLoss()

    def forward(self, depth_pseudo, depth_hq):
        return self.l1(depth_pseudo, depth_hq) + self.lambda_grad * self.grad(depth_pseudo, depth_hq)


class StageILoss(nn.Module):
    """L_stage1 = L1(pixel) + lambda_1 * L_depth."""
    def __init__(self, lambda_depth: float = 0.1, lambda_grad: float = 0.5):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.l1 = nn.L1Loss()
        self.depth_loss = EdgeAwareDepthLoss(lambda_grad=lambda_grad)

    def forward(self, x_hq, x_gt, depth_pseudo, depth_hq):
        l_pix = self.l1(x_hq, x_gt)
        l_dep = self.depth_loss(depth_pseudo, depth_hq)
        total = l_pix + self.lambda_depth * l_dep
        return total, {"pix": l_pix.detach(), "depth": l_dep.detach()}


class StageIILoss(nn.Module):
    """L_stage2 = L1(pixel) + L_diff."""
    def __init__(self, lambda_diff: float = 1.0, lambda_eps: float = 0.0):
        """`lambda_eps` enables an optional auxiliary noise-prediction loss."""
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_eps = lambda_eps
        self.l1 = nn.L1Loss()

    def forward(self, x_hq, x_gt, Z, Z_hat, eps_pred=None, eps=None):
        l_pix = self.l1(x_hq, x_gt)
        l_diff = self.l1(Z, Z_hat)
        total = l_pix + self.lambda_diff * l_diff
        log = {"pix": l_pix.detach(), "diff": l_diff.detach()}
        if self.lambda_eps > 0 and eps_pred is not None and eps is not None:
            l_eps = self.l1(eps_pred, eps)
            total = total + self.lambda_eps * l_eps
            log["eps"] = l_eps.detach()
        return total, log
