import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    targets = targets.float()
    inter = (probs * targets).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _tversky_loss(probs: torch.Tensor, targets: torch.Tensor,
                  alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:

    targets = targets.float()
    TP = (probs * targets).sum(dim=(0, 2, 3))
    FP = (probs * (1.0 - targets)).sum(dim=(0, 2, 3))
    FN = ((1.0 - probs) * targets).sum(dim=(0, 2, 3))
    t = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return 1.0 - t.mean()


class ComboLoss(nn.Module):
    """
    ComboLoss = w_bce * BCE  +  w_dice * Dice  +  w_tversky * Tversky
    """
    def __init__(
        self,
        pos_weight: float = 400.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        weights: tuple[float, float, float] = (1.0, 2.0, 1.2),
        from_logits: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.w_bce, self.w_dice, self.w_tversky = map(float, weights)
        self.from_logits = bool(from_logits)
        self.eps = float(eps)

        if self.from_logits:
            # single-class segmentation -> scalar pos_weight is fine
            self.register_buffer("posw_tensor", torch.tensor([self.pos_weight]))
            self.bce_logits = nn.BCEWithLogitsLoss(pos_weight=self.posw_tensor)

    def _bce_from_probs(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(self.eps, 1.0 - self.eps)
        targets = targets.float()
        pos_term = -targets * torch.log(probs)
        neg_term = -(1.0 - targets) * torch.log(1.0 - probs)
        # weight positive pixels
        loss = self.pos_weight * pos_term + neg_term
        return loss.mean()

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            bce = self.bce_logits(x, targets)
            probs = torch.sigmoid(x)
        else:
            probs = x
            bce = self._bce_from_probs(probs, targets)

        dice = _soft_dice_loss(probs, targets, eps=self.eps)
        tversky = _tversky_loss(probs, targets, alpha=self.alpha, beta=self.beta, eps=self.eps)
        return self.w_bce * bce + self.w_dice * dice + self.w_tversky * tversky
