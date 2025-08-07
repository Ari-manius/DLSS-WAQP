import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing severe class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    
    Args:
        alpha (float or tensor): Weighting factor for rare class (alpha=1 means no weighting)
        gamma (float): Focusing parameter to down-weight easy examples
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where N is batch size and C is number of classes
            targets: (N,) where each value is 0 <= targets[i] <= C-1
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss that automatically computes alpha weights 
    based on effective number of samples.
    
    Paper: "Class-Balanced Loss Based on Effective Number of Samples" 
    (https://arxiv.org/abs/1901.05555)
    """
    def __init__(self, beta=0.9999, gamma=2.0, reduction='mean'):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, samples_per_class=None):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class labels
            samples_per_class: (C,) number of samples per class
        """
        if samples_per_class is None:
            # Calculate from targets
            unique, counts = torch.unique(targets, return_counts=True)
            samples_per_class = torch.zeros(inputs.size(1), device=inputs.device)
            samples_per_class[unique] = counts.float()

        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * len(weights)  # Normalize

        # Apply focal loss with class balancing
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get class weights for each sample
        alpha_t = weights[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with automatic class weight calculation based on inverse frequency.
    """
    def __init__(self, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class labels
        """
        # Calculate class frequencies
        unique, counts = torch.unique(targets, return_counts=True)
        class_weights = torch.zeros(inputs.size(1), device=inputs.device)
        
        # Inverse frequency weighting
        total_samples = len(targets)
        for i, class_idx in enumerate(unique):
            class_weights[class_idx] = total_samples / (len(unique) * counts[i])

        return F.cross_entropy(inputs, targets, weight=class_weights, reduction=self.reduction)


def get_class_weights(targets, method='inverse_freq', beta=0.9999):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        targets: (N,) tensor of class labels
        method: 'inverse_freq', 'effective_num', or 'sqrt_inv_freq'
        beta: parameter for effective number calculation
    
    Returns:
        torch.Tensor: class weights
    """
    unique, counts = torch.unique(targets, return_counts=True)
    num_classes = len(unique)
    weights = torch.zeros(num_classes, device=targets.device)
    
    if method == 'inverse_freq':
        total_samples = len(targets)
        for i, class_idx in enumerate(unique):
            weights[class_idx] = total_samples / (num_classes * counts[i])
            
    elif method == 'effective_num':
        for i, class_idx in enumerate(unique):
            effective_num = 1.0 - (beta ** counts[i])
            weights[class_idx] = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        
    elif method == 'sqrt_inv_freq':
        total_samples = len(targets)
        for i, class_idx in enumerate(unique):
            weights[class_idx] = np.sqrt(total_samples / counts[i])
        weights = weights / weights.sum() * num_classes
        
    return weights


def compute_focal_alpha(targets, method='inverse_freq'):
    """
    Compute alpha parameter for Focal Loss based on class distribution.
    
    Args:
        targets: (N,) tensor of class labels
        method: method for computing alpha weights
    
    Returns:
        torch.Tensor: alpha weights for each class
    """
    unique, counts = torch.unique(targets, return_counts=True)
    num_classes = len(unique)
    alpha = torch.ones(num_classes, device=targets.device)
    
    if method == 'inverse_freq':
        total_samples = len(targets)
        min_count = counts.min().float()
        
        for i, class_idx in enumerate(unique):
            alpha[class_idx] = min_count / counts[i]
            
    return alpha