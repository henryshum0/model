from torch import nn, Tensor
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, beta=1.0, reduction='mean'):
        """
        Focal Loss for dense pixel-wise binary classification

        Args:
            alpha: Weight factor for class balance. Must be in [0,1].
                  alpha is for positive class (crack pixels),
                  (1-alpha) is for negative class (background pixels)
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted
            beta: Global scaling factor for the focal term
            reduction: 'mean', 'sum' or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, inputs, targets):
        # Apply sigmoid to raw logits
        probs = torch.sigmoid(inputs)

        # Binary cross entropy
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Apply the focal term
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.beta * ((1 - pt) ** self.gamma)

        # Apply balanced alpha weighting:
        # For positive examples (targets==1): use alpha
        # For negative examples (targets==0): use (1-alpha)
        if 0 <= self.alpha <= 1:  # Ensure alpha is in valid range
            alpha_weight = torch.ones_like(targets) * (1 - self.alpha)  # Default for negative class
            alpha_weight = torch.where(targets == 1, torch.ones_like(targets) * self.alpha, alpha_weight)  # Set alpha for positive class
            focal_weight = alpha_weight * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        
class DiceLoss(nn.Module):
    def __init__(self, epsilon:float=1e-6, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        
    def forward(self, input:Tensor, target:Tensor):
        # input: (B, 1, H, W) or (B, H, W)
        # target: (B, 1, H, W) or (B, H, W)
        input = torch.sigmoid(input)
        if input.dim() == 4:
            input = input.view(input.size(0), -1)
            target = target.view(target.size(0), -1)
        elif input.dim() == 3:
            input = input.view(input.size(0), -1)
            target = target.view(target.size(0), -1)
        else:
            raise ValueError("Input and target must be 3D or 4D tensors")
        
        intersection = (2 * input * target).sum(dim=1)
        denominator = input.sum(dim=1) + target.sum(dim=1) + self.epsilon
        dice = (intersection + self.epsilon) / denominator
        loss = 1 - dice

        if self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()
    