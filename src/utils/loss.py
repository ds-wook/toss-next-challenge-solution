import torch
import torch.nn as nn
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 0.5, reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Compute BCE loss with logits for numerical stability
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Compute pt (probability of true class)
        pt = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, pt, 1 - pt)

        # Compute alpha_t (class-specific alpha)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Compute focal weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for CTR prediction"""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Compute BCE loss with logits for numerical stability
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Compute pt (probability of true class)
        pt = torch.sigmoid(inputs)
        pt = torch.clamp(pt, self.clip, 1 - self.clip)

        # Compute asymmetric weights
        pt_neg = torch.where(targets == 1, 1 - pt, pt)
        pt_pos = torch.where(targets == 1, pt, 1 - pt)

        # Apply asymmetric focal weights
        focal_weight_neg = (1 - pt_neg) ** self.gamma_neg
        focal_weight_pos = (1 - pt_pos) ** self.gamma_pos

        # Apply weights
        asymmetric_loss = focal_weight_neg * focal_weight_pos * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return asymmetric_loss.mean()
        elif self.reduction == "sum":
            return asymmetric_loss.sum()
        else:
            return asymmetric_loss


class LabelSmoothingBCE(nn.Module):
    """Binary Cross Entropy with Label Smoothing"""

    def __init__(
        self, smoothing: float = 0.1, pos_weight: float = 1.0, reduction: str = "mean"
    ):
        super(LabelSmoothingBCE, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Apply label smoothing
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        # Compute BCE loss with smoothed targets
        loss = nn.functional.binary_cross_entropy_with_logits(
            inputs,
            smoothed_targets,
            pos_weight=torch.tensor(self.pos_weight),
            reduction=self.reduction,
        )
        return loss


class ClassSpecificLRFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 0.5,
        pos_lr_multiplier: float = 5.0,
        neg_lr_multiplier: float = 1.0,
        reduction: str = "mean",
    ):
        super(ClassSpecificLRFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_lr_multiplier = pos_lr_multiplier
        self.neg_lr_multiplier = neg_lr_multiplier
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Compute BCE loss with logits for numerical stability
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Compute pt (probability of true class)
        pt = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, pt, 1 - pt)

        # Compute alpha_t (class-specific alpha)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Compute focal weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply class-specific learning rate multipliers
        pos_mask = targets == 1
        neg_mask = targets == 0

        scaled_losses = torch.zeros_like(focal_loss)
        scaled_losses[pos_mask] = focal_loss[pos_mask] * self.pos_lr_multiplier
        scaled_losses[neg_mask] = focal_loss[neg_mask] * self.neg_lr_multiplier

        # Apply reduction
        if self.reduction == "mean":
            return scaled_losses.mean()
        elif self.reduction == "sum":
            return scaled_losses.sum()
        else:
            return scaled_losses


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Get loss function based on name"""
    if loss_name.lower() == "bce_with_logits":
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(kwargs.get("pos_weight", 1.0), dtype=torch.float32)
        )
    elif loss_name.lower() == "focal":
        return ClassSpecificLRFocalLoss(
            alpha=kwargs.get("alpha", 1),
            gamma=kwargs.get("focal_gamma", 0.5),
            pos_lr_multiplier=kwargs.get("pos_lr_multiplier", 5),
            neg_lr_multiplier=kwargs.get("neg_lr_multiplier", 1),
        )
    elif loss_name.lower() == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=kwargs.get("gamma_neg", 4.0),
            gamma_pos=kwargs.get("gamma_pos", 1.0),
            clip=kwargs.get("clip", 0.05),
        )
    elif loss_name.lower() == "label_smoothing":
        pos_weight = kwargs.get("pos_weight", 1.0)
        pos_weight = min(pos_weight, 10.0)  # Cap at 10x
        return LabelSmoothingBCE(
            smoothing=kwargs.get("smoothing", 0.1),
            pos_weight=pos_weight,
        )
    else:
        raise ValueError(
            f"Unsupported loss function: {loss_name}. Choose from ['bce_with_logits', 'focal', 'asymmetric', 'label_smoothing']"
        )


# Compute alpha based on current batch
def compute_focal_loss_alpha(targets: Tensor) -> float:
    pos_count = targets.sum()
    neg_count = len(targets) - pos_count
    alpha = neg_count / len(targets)
    return alpha
