import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.utils import to_networkx
import networkx as nx
from collections import defaultdict
import copy


def graph_aware_split(data, n_splits=5, test_size=0.2, random_state=42, 
                     preserve_connectivity=True, stratify=True):
    """
    Graph-aware train/validation/test split that preserves graph structure.
    
    Args:
        data: PyTorch Geometric Data object
        n_splits: Number of CV folds
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        preserve_connectivity: Whether to ensure splits preserve graph connectivity
        stratify: Whether to stratify splits by class labels (for classification)
    
    Returns:
        List of (train_mask, val_mask, test_mask) tuples for each fold
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    num_nodes = data.num_nodes
    node_indices = np.arange(num_nodes)
    
    # First, create test set
    test_size_nodes = int(num_nodes * test_size)
    
    if stratify and data.y is not None and len(data.y.unique()) > 1:
        # Stratified split for classification
        from sklearn.model_selection import train_test_split
        
        train_val_indices, test_indices = train_test_split(
            node_indices, test_size=test_size, 
            stratify=data.y.cpu().numpy(), 
            random_state=random_state
        )
    else:
        # Random split for regression or unstratified
        np.random.shuffle(node_indices)
        test_indices = node_indices[:test_size_nodes]
        train_val_indices = node_indices[test_size_nodes:]
    
    # Create test mask
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True
    
    # Now create CV splits for train/validation
    splits = []
    
    if stratify and data.y is not None and len(data.y.unique()) > 1:
        # Stratified K-Fold for classification
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_val_labels = data.y[train_val_indices].cpu().numpy()
        split_generator = skf.split(train_val_indices, train_val_labels)
    else:
        # Regular K-Fold for regression
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_generator = kf.split(train_val_indices)
    
    for train_idx, val_idx in split_generator:
        # Convert to global indices
        global_train_idx = train_val_indices[train_idx]
        global_val_idx = train_val_indices[val_idx]
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[global_train_idx] = True
        val_mask[global_val_idx] = True
        
        # Ensure connectivity if requested
        if preserve_connectivity:
            train_mask, val_mask = ensure_connectivity(data, train_mask, val_mask)
        
        splits.append((train_mask, val_mask, test_mask))
    
    return splits


def ensure_connectivity(data, train_mask, val_mask):
    """
    Ensure that train and validation sets maintain graph connectivity.
    
    Args:
        data: PyTorch Geometric Data object
        train_mask: Boolean mask for training nodes
        val_mask: Boolean mask for validation nodes
    
    Returns:
        Adjusted train_mask and val_mask that preserve connectivity
    """
    # Convert to NetworkX for connectivity analysis
    G = to_networkx(data, to_undirected=True)
    
    # Get connected components
    connected_components = list(nx.connected_components(G))
    
    # For each connected component, ensure both train and val have nodes
    train_indices = torch.where(train_mask)[0].tolist()
    val_indices = torch.where(val_mask)[0].tolist()
    
    adjusted_train = set(train_indices)
    adjusted_val = set(val_indices)
    
    for component in connected_components:
        component_train = component.intersection(adjusted_train)
        component_val = component.intersection(adjusted_val)
        
        # If component has nodes only in one set, move some to the other
        if len(component_train) == 0 and len(component_val) > 1:
            # Move one node from val to train
            node_to_move = list(component_val)[0]
            adjusted_val.remove(node_to_move)
            adjusted_train.add(node_to_move)
        elif len(component_val) == 0 and len(component_train) > 1:
            # Move one node from train to val
            node_to_move = list(component_train)[0]
            adjusted_train.remove(node_to_move)
            adjusted_val.add(node_to_move)
    
    # Create new masks
    new_train_mask = torch.zeros_like(train_mask)
    new_val_mask = torch.zeros_like(val_mask)
    
    new_train_mask[list(adjusted_train)] = True
    new_val_mask[list(adjusted_val)] = True
    
    return new_train_mask, new_val_mask


def cross_validate_model(model_class, data, model_params, train_params, 
                        n_splits=5, test_size=0.2, random_state=42, 
                        device=torch.device('cpu'), verbose=True):
    """
    Perform cross-validation for a GNN model.
    
    Args:
        model_class: Model class to instantiate
        data: PyTorch Geometric Data object
        model_params: Dictionary of model parameters
        train_params: Dictionary of training parameters
        n_splits: Number of CV folds
        test_size: Test set size
        random_state: Random seed
        device: Device to use for training
        verbose: Whether to print progress
    
    Returns:
        Dictionary with CV results
    """
    # Get splits
    splits = graph_aware_split(data, n_splits=n_splits, test_size=test_size, 
                              random_state=random_state)
    
    # Store results
    fold_results = []
    
    for fold, (train_mask, val_mask, test_mask) in enumerate(splits):
        if verbose:
            print(f"\nFold {fold + 1}/{n_splits}")
            print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        
        # Create data copy for this fold
        fold_data = copy.deepcopy(data)
        fold_data.train_mask = train_mask
        fold_data.val_mask = val_mask
        fold_data.test_mask = test_mask
        
        # Initialize model
        input_dim = fold_data.num_node_features
        output_dim = len(fold_data.y.unique()) if len(fold_data.y.unique()) > 1 else 1
        
        model = model_class(input_dim=input_dim, output_dim=output_dim, **model_params)
        model = model.to(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), **train_params.get('optimizer', {}))
        
        if len(fold_data.y.unique()) > 1:  # Classification
            criterion = torch.nn.CrossEntropyLoss()
        else:  # Regression
            criterion = torch.nn.L1Loss()
        
        # Training loop (simplified)
        model.train()
        fold_data = fold_data.to(device)
        
        train_losses = []
        val_losses = []
        
        epochs = train_params.get('epochs', 100)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(fold_data)
            train_loss = criterion(output[train_mask], fold_data.y[train_mask])
            train_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                output = model(fold_data)
                val_loss = criterion(output[val_mask], fold_data.y[val_mask])
            model.train()
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            if epoch % 20 == 0 and verbose:
                print(f"  Epoch {epoch}: Train Loss {train_loss.item():.4f}, "
                      f"Val Loss {val_loss.item():.4f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            output = model(fold_data)
            
            # Test performance
            test_loss = criterion(output[test_mask], fold_data.y[test_mask]).item()
            
            if len(fold_data.y.unique()) > 1:  # Classification
                _, test_pred = torch.max(output[test_mask], 1)
                test_acc = (test_pred == fold_data.y[test_mask]).float().mean().item()
                
                # Validation accuracy
                _, val_pred = torch.max(output[val_mask], 1)
                val_acc = (val_pred == fold_data.y[val_mask]).float().mean().item()
                
                fold_result = {
                    'fold': fold,
                    'val_loss': val_losses[-1],
                    'val_accuracy': val_acc,
                    'test_loss': test_loss,
                    'test_accuracy': test_acc,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
            else:  # Regression
                fold_result = {
                    'fold': fold,
                    'val_loss': val_losses[-1],
                    'test_loss': test_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
        
        fold_results.append(fold_result)
        
        if verbose:
            if len(fold_data.y.unique()) > 1:
                print(f"  Final - Val Acc: {fold_result['val_accuracy']:.4f}, "
                      f"Test Acc: {fold_result['test_accuracy']:.4f}")
            else:
                print(f"  Final - Val Loss: {fold_result['val_loss']:.4f}, "
                      f"Test Loss: {fold_result['test_loss']:.4f}")
    
    # Aggregate results
    if len(data.y.unique()) > 1:  # Classification
        val_accs = [r['val_accuracy'] for r in fold_results]
        test_accs = [r['test_accuracy'] for r in fold_results]
        
        results = {
            'fold_results': fold_results,
            'mean_val_accuracy': np.mean(val_accs),
            'std_val_accuracy': np.std(val_accs),
            'mean_test_accuracy': np.mean(test_accs),
            'std_test_accuracy': np.std(test_accs),
            'val_accuracies': val_accs,
            'test_accuracies': test_accs
        }
    else:  # Regression
        val_losses = [r['val_loss'] for r in fold_results]
        test_losses = [r['test_loss'] for r in fold_results]
        
        results = {
            'fold_results': fold_results,
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'mean_test_loss': np.mean(test_losses),
            'std_test_loss': np.std(test_losses),
            'val_losses': val_losses,
            'test_losses': test_losses
        }
    
    return results


def print_cv_results(results):
    """Print cross-validation results in a formatted way."""
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    
    if 'mean_val_accuracy' in results:  # Classification
        print(f"Validation Accuracy: {results['mean_val_accuracy']:.4f} ± {results['std_val_accuracy']:.4f}")
        print(f"Test Accuracy:       {results['mean_test_accuracy']:.4f} ± {results['std_test_accuracy']:.4f}")
        print(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in results['test_accuracies']]}")
    else:  # Regression
        print(f"Validation Loss: {results['mean_val_loss']:.4f} ± {results['std_val_loss']:.4f}")
        print(f"Test Loss:       {results['mean_test_loss']:.4f} ± {results['std_test_loss']:.4f}")
        print(f"Individual fold losses: {[f'{loss:.4f}' for loss in results['test_losses']]}")
    
    print("="*50)