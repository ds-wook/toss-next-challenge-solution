import torch
import math


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing Learning Rate Scheduler"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch: int):
        """Update learning rate for given epoch"""
        if epoch < self.warmup_epochs:
            # Warmup phase: linear increase from warmup_start_lr to base_lr
            lr = (
                self.warmup_start_lr
                + (self.base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
            )
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class WarmupLinearScheduler:
    """Warmup + Linear Decay Learning Rate Scheduler"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch: int):
        """Update learning rate for given epoch"""
        if epoch < self.warmup_epochs:
            # Warmup phase: linear increase from warmup_start_lr to base_lr
            lr = (
                self.warmup_start_lr
                + (self.base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
            )
        else:
            # Linear decay phase
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.base_lr - (self.base_lr - self.min_lr) * progress

        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


def get_scheduler(
    scheduler_name: str, optimizer: torch.optim.Optimizer, total_epochs: int, **kwargs
) -> object:
    """Get learning rate scheduler based on name"""

    if scheduler_name.lower() == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", max(1, total_epochs // 10)),
            total_epochs=total_epochs,
            min_lr=kwargs.get("min_lr", 1e-6),
            warmup_start_lr=kwargs.get("warmup_start_lr", 1e-7),
        )
    elif scheduler_name.lower() == "warmup_linear":
        return WarmupLinearScheduler(
            optimizer=optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", max(1, total_epochs // 10)),
            total_epochs=total_epochs,
            min_lr=kwargs.get("min_lr", 1e-6),
            warmup_start_lr=kwargs.get("warmup_start_lr", 1e-7),
        )
    elif scheduler_name.lower() == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.7),
            patience=kwargs.get("patience", 3),
            min_lr=kwargs.get("min_lr", 1e-6),
            verbose=True,
            threshold=kwargs.get("threshold", 1e-4),
        )
    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. Choose from ['warmup_cosine', 'warmup_linear', 'reduce_on_plateau']"
        )
