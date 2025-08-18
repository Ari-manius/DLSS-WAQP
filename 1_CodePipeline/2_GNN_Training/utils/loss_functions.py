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
    Enhanced Class-Balanced Focal Loss with improved smallest class handling.
    
    Paper: "Class-Balanced Loss Based on Effective Number of Samples" 
    (https://arxiv.org/abs/1901.05555)
    """
    def __init__(self, beta=0.9999, gamma=2.0, reduction='mean', min_class_boost=2.0):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.min_class_boost = min_class_boost  # Extra boost for smallest class

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
        effective_num = torch.clamp(effective_num, min=1e-7)  # Avoid division by zero
        weights = (1.0 - self.beta) / effective_num
        
        # Boost the smallest class(es) further
        min_samples = samples_per_class[samples_per_class > 0].min()
        smallest_class_mask = (samples_per_class == min_samples) & (samples_per_class > 0)
        weights[smallest_class_mask] *= self.min_class_boost
        
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
        min_count = counts.min().float()
        
        for i, class_idx in enumerate(unique):
            alpha[class_idx] = min_count / counts[i]
            
    return alpha


def smart_oversample_indices(train_indices, train_labels, strategy='balanced', min_samples_factor=3, max_total_samples=None):
    """
    Create oversampling indices with focus on minority classes and memory constraints.
    
    Args:
        train_indices: tensor of training node indices
        train_labels: tensor of training labels
        strategy: 'balanced' (equal samples) or 'boosted' (boost minority more)
        min_samples_factor: minimum factor to multiply smallest class
        max_total_samples: maximum total samples to prevent memory issues
    
    Returns:
        tensor of oversampled indices
    """
    unique_classes, class_counts = torch.unique(train_labels, return_counts=True)
    original_total = len(train_indices)
    
    # Set memory-safe default if not specified
    if max_total_samples is None:
        max_total_samples = min(original_total * 5, 50000)  # Cap at 5x original or 50k
    
    if strategy == 'balanced':
        target_count = class_counts.max()
    elif strategy == 'boosted':
        # Boost smallest classes more aggressively
        min_count = class_counts.min()
        target_count = min(class_counts.max(), min_count * min_samples_factor)
    
    # Calculate total samples if we use target_count for all classes
    projected_total = target_count * len(unique_classes)
    
    # Scale down if projected total exceeds memory limit
    if projected_total > max_total_samples:
        scale_factor = max_total_samples / projected_total
        target_count = int(target_count * scale_factor)
        print(f"Scaling down oversampling: target_count reduced to {target_count} to stay within memory limit")
    
    oversampled_indices = []
    
    for class_idx in unique_classes:
        class_mask = train_labels == class_idx
        class_indices = train_indices[class_mask]
        current_count = len(class_indices)
        
        if current_count == 0:
            continue
            
        # Determine how many samples we need
        if strategy == 'balanced':
            needed_samples = target_count
        else:  # boosted
            # Give smallest classes more samples
            boost_factor = min_count / current_count if current_count > 0 else 1
            needed_samples = int(current_count * max(1, boost_factor * min_samples_factor))
            needed_samples = min(needed_samples, target_count)
        
        # Additional safety check per class
        needed_samples = min(needed_samples, max_total_samples // len(unique_classes))
        
        # Generate indices with repetition
        n_repeats = needed_samples // current_count
        remainder = needed_samples % current_count
        
        repeated = class_indices.repeat(n_repeats)
        if remainder > 0:
            perm_indices = torch.randperm(current_count, device=class_indices.device)[:remainder]
            repeated = torch.cat([repeated, class_indices[perm_indices]])
        
        oversampled_indices.append(repeated)
    
    final_indices = torch.cat(oversampled_indices)
    
    # Final safety check
    if len(final_indices) > max_total_samples:
        perm = torch.randperm(len(final_indices), device=final_indices.device)[:max_total_samples]
        final_indices = final_indices[perm]
        print(f"Final safety trim: reduced to {len(final_indices)} samples")
    
    return final_indices