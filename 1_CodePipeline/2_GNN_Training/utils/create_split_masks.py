import torch 

def create_split_masks(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    torch.manual_seed(seed)
    class_indices = [torch.where(data.y == c)[0] for c in torch.unique(data.y)]
    
    train_idx, val_idx, test_idx = [], [], []

    for indices in class_indices:
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        indices = indices[torch.randperm(n)]
        train_idx.append(indices[:n_train])
        val_idx.append(indices[n_train:n_train + n_val])
        test_idx.append(indices[n_train + n_val:])

    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)
    test_idx = torch.cat(test_idx)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

def create_split_masks_regression(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    torch.manual_seed(seed)
    n = data.num_nodes
    indices = torch.randperm(n)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask

def create_cv_splits_from_run_id(data, run_id, n_folds=3, train_ratio=0.8, seed_base=42, use_kfold=False):
    """
    Create different data splits for each run to enable cross-validation across runs.
    
    Args:
        data: PyTorch Geometric data object
        run_id: String like "run1", "run2", "run3" 
        n_folds: Total number of folds/runs expected
        train_ratio: Proportion for training (rest split between val/test) - only used if not use_kfold
        seed_base: Base seed for reproducibility
        use_kfold: If True, uses proper k-fold CV where each run tests on a different fold
    
    Returns:
        train_mask, val_mask, test_mask for this specific run
    """
    # Extract run number from run_id (e.g., "run1" -> 1)
    if isinstance(run_id, str) and run_id.startswith('run'):
        run_num = int(run_id[3:])  # Extract number after 'run'
    else:
        run_num = 1  # Default fallback
    
    torch.manual_seed(seed_base)  # Same seed for consistent fold creation
    n = data.num_nodes
    
    if use_kfold and n_folds > 1:
        # True k-fold cross-validation: each run tests on a different fold
        indices = torch.randperm(n)
        fold_size = n // n_folds
        
        # Calculate test indices for this fold
        test_start = (run_num - 1) * fold_size
        test_end = run_num * fold_size if run_num < n_folds else n
        test_idx = indices[test_start:test_end]
        
        # Remaining indices for train/val
        remaining_idx = torch.cat([indices[:test_start], indices[test_end:]])
        
        # Split remaining into train/val (e.g., 80/20)
        n_remaining = len(remaining_idx)
        n_train = int(n_remaining * 0.8)
        
        train_idx = remaining_idx[:n_train]
        val_idx = remaining_idx[n_train:]
        
        print(f"🎯 K-fold CV Run {run_num}: Testing on fold {run_num}/{n_folds}")
        
    else:
        # Different random splits for each run (original approach)
        run_seed = seed_base + (run_num - 1) * 100
        torch.manual_seed(run_seed)
        indices = torch.randperm(n)
        
        n_train = int(n * train_ratio)
        n_remaining = n - n_train
        n_val = n_remaining // 2
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        print(f"🎲 Random split Run {run_num} (seed={run_seed})")

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    print(f"   📊 Split sizes: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")
    
    return train_mask, val_mask, test_mask
