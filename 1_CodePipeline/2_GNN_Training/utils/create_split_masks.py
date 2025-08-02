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
