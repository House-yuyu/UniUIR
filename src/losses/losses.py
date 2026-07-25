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
    """Paper Eq. (4): L_stage2 = L1(pixel) + L1(Z, Z_hat)."""
    def __init__(
        self,
        lambda_pix: float = 1.0,
        lambda_diff: float = 1.0,
        lambda_eps: float = 0.0,
        lambda_teacher: float = 0.0,
        lambda_teacher_mse: float = 0.0,
        lambda_mse: float = 0.0,
        lambda_log_mse: float = 0.0,
        lambda_direct_diff: float = 0.0,
        latent_scale: float = 64.0,
    ):
        """`lambda_eps` enables an optional auxiliary noise-prediction loss."""
        super().__init__()
        self.lambda_pix = lambda_pix
        self.lambda_diff = lambda_diff
        self.lambda_eps = lambda_eps
        self.lambda_teacher = lambda_teacher
        self.lambda_teacher_mse = lambda_teacher_mse
        self.lambda_mse = lambda_mse
        self.lambda_log_mse = lambda_log_mse
        self.lambda_direct_diff = lambda_direct_diff
        if latent_scale <= 0:
            raise ValueError(f"latent_scale must be positive, got {latent_scale}")
        self.latent_scale = latent_scale
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def prior_loss(
        self,
        Z,
        Z_hat,
        eps_pred=None,
        eps=None,
        Z_direct=None,
    ):

        l_diff = self.l1(Z, Z_hat)
        total = self.lambda_diff * l_diff
        log = {"diff": l_diff.detach()}
        if self.lambda_direct_diff > 0 and Z_direct is not None:
            l_direct_diff = self.l1(Z, Z_direct)
            total = total + self.lambda_direct_diff * l_direct_diff
            log["direct_diff"] = l_direct_diff.detach()
        if self.lambda_eps > 0 and eps_pred is not None and eps is not None:
            l_eps = self.l1(eps_pred, eps)
            total = total + self.lambda_eps * l_eps
            log["eps"] = l_eps.detach()
        return total, log

    def forward(
        self,
        x_hq,
        x_gt,
        Z,
        Z_hat,
        eps_pred=None,
        eps=None,
        x_teacher=None,
        Z_direct=None,
    ):
        l_pix = self.l1(x_hq, x_gt)
        prior_total, log = self.prior_loss(
            Z,
            Z_hat,
            eps_pred=eps_pred,
            eps=eps,
            Z_direct=Z_direct,
        )
        total = self.lambda_pix * l_pix + prior_total
        log["pix"] = l_pix.detach()
        if self.lambda_mse > 0:
            l_mse = self.mse(x_hq, x_gt)
            total = total + self.lambda_mse * l_mse
            log["mse"] = l_mse.detach()
        if self.lambda_log_mse > 0:
            per_image_mse = (x_hq - x_gt).square().flatten(1).mean(dim=1)
            l_log_mse = torch.log(per_image_mse.clamp_min(1e-8)).mean()
            total = total + self.lambda_log_mse * l_log_mse
            log["log_mse"] = l_log_mse.detach()
        if self.lambda_teacher > 0 and x_teacher is not None:
            l_teacher = self.l1(x_hq, x_teacher)
            total = total + self.lambda_teacher * l_teacher
            log["teacher"] = l_teacher.detach()
        if self.lambda_teacher_mse > 0 and x_teacher is not None:
            l_teacher_mse = self.mse(x_hq, x_teacher)
            total = total + self.lambda_teacher_mse * l_teacher_mse
            log["teacher_mse"] = l_teacher_mse.detach()
        return total, log
