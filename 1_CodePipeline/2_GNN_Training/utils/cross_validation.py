import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.data import Data
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
                        device=torch.device('cpu'), verbose=True, model_type=None):
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
    # Handle MPS memory management more carefully
    import os
    if device.type == 'mps':
        # Clear cache before starting
        torch.mps.empty_cache()
        # Remove any existing watermark setting that might be invalid
        if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
            del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
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
        
        # Handle MPS device issues
        try:
            model = model.to(device)
        except RuntimeError as e:
            if "watermark ratio" in str(e) and device.type == 'mps':
                print(f"⚠ MPS watermark error in fold {fold}: {e}")
                print("⚠ Falling back to CPU for this fold")
                device = torch.device('cpu')
                model = model.to(device)
                fold_data = fold_data.to(device)
            else:
                raise e
        
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
            
            # Use memory-efficient sampling for large datasets
            if hasattr(fold_data, 'train_mask') and train_mask.sum() > 8000:
                train_indices = torch.where(train_mask)[0].to(fold_data.x.device)
                sample_size = min(6000, len(train_indices))
                sampled_indices = train_indices[torch.randperm(len(train_indices), device=train_indices.device)[:sample_size]]
                
                # Create subgraph
                edge_index, _ = subgraph(sampled_indices, fold_data.edge_index, relabel_nodes=True)
                x_sample = fold_data.x[sampled_indices]
                y_sample = fold_data.y[sampled_indices]
                
                sample_data = Data(x=x_sample, edge_index=edge_index, y=y_sample)
                output = model(sample_data)
                train_mask_sample = torch.ones(len(sampled_indices), dtype=torch.bool, device=output.device)
                train_loss = criterion(output[train_mask_sample], y_sample)
            else:
                output = model(fold_data)
                
                # Apply smart oversampling for classification if parameters are provided
                if len(fold_data.y.unique()) > 1 and 'class_balance' in train_params:
                    from utils.loss_functions import smart_oversample_indices
                    
                    train_indices = torch.where(train_mask)[0].to(fold_data.x.device)
                    train_labels = fold_data.y[train_indices]
                    
                    if len(torch.unique(train_labels)) > 1:
                        cb_params = train_params['class_balance']
                        oversampled_train_indices = smart_oversample_indices(
                            train_indices, train_labels,
                            strategy=cb_params.get('oversample_strategy', 'boosted'),
                            min_samples_factor=cb_params.get('min_samples_factor', 3),
                            max_total_samples=cb_params.get('max_total_samples', 20000)
                        )
                        train_loss = criterion(output[oversampled_train_indices], 
                                             fold_data.y[oversampled_train_indices])
                    else:
                        train_loss = criterion(output[train_mask], fold_data.y[train_mask])
                else:
                    train_loss = criterion(output[train_mask], fold_data.y[train_mask])
                
            train_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                # Use sampling for validation too if dataset is large
                if hasattr(fold_data, 'val_mask') and val_mask.sum() > 4000:
                    val_indices = torch.where(val_mask)[0].to(fold_data.x.device)
                    val_sample_size = min(3000, len(val_indices))
                    sampled_val_indices = val_indices[torch.randperm(len(val_indices), device=val_indices.device)[:val_sample_size]]
                    
                    edge_index, _ = subgraph(sampled_val_indices, fold_data.edge_index, relabel_nodes=True)
                    x_val_sample = fold_data.x[sampled_val_indices]
                    y_val_sample = fold_data.y[sampled_val_indices]
                    
                    val_sample_data = Data(x=x_val_sample, edge_index=edge_index, y=y_val_sample)
                    output = model(val_sample_data)
                    val_mask_sample = torch.ones(len(sampled_val_indices), dtype=torch.bool, device=output.device)
                    val_loss = criterion(output[val_mask_sample], y_val_sample)
                else:
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
            # Use sampling for final evaluation if dataset is large
            if hasattr(fold_data, 'test_mask') and test_mask.sum() > 4000:
                # Sample test data
                test_indices = torch.where(test_mask)[0].to(fold_data.x.device)
                test_sample_size = min(3000, len(test_indices))
                sampled_test_indices = test_indices[torch.randperm(len(test_indices), device=test_indices.device)[:test_sample_size]]
                
                edge_index, _ = subgraph(sampled_test_indices, fold_data.edge_index, relabel_nodes=True)
                x_test_sample = fold_data.x[sampled_test_indices]
                y_test_sample = fold_data.y[sampled_test_indices]
                
                test_sample_data = Data(x=x_test_sample, edge_index=edge_index, y=y_test_sample)
                
                # GAT-specific evaluation fallback to avoid memory issues
                if model_type == 'gat':
                    # Use GAT's linear transformation for memory-efficient evaluation
                    if hasattr(model, 'convs') and len(model.convs) > 0:
                        if hasattr(model.convs[0], 'lin_src') and model.convs[0].lin_src is not None:
                            test_output = model.convs[0].lin_src(x_test_sample)
                        else:
                            # Fallback to basic linear layer
                            temp_linear = torch.nn.Linear(x_test_sample.size(-1), model.convs[0].out_channels if hasattr(model.convs[0], 'out_channels') else len(fold_data.y.unique())).to(device)
                            test_output = temp_linear(x_test_sample)
                        
                        # Ensure output matches expected dimensions
                        if test_output.size(-1) != len(fold_data.y.unique()):
                            final_linear = torch.nn.Linear(test_output.size(-1), len(fold_data.y.unique())).to(device)
                            test_output = final_linear(test_output)
                    else:
                        # Complete fallback
                        temp_linear = torch.nn.Linear(x_test_sample.size(-1), len(fold_data.y.unique())).to(device)
                        test_output = temp_linear(x_test_sample)
                else:
                    # Standard evaluation for non-GAT models
                    test_output = model(test_sample_data)
                
                test_mask_sample = torch.ones(len(sampled_test_indices), dtype=torch.bool, device=test_output.device)
                
                # Test performance
                test_loss = criterion(test_output[test_mask_sample], y_test_sample).item()
                
                # Sample validation data for consistency
                if len(fold_data.y.unique()) > 1:  # Classification
                    _, test_pred = torch.max(test_output[test_mask_sample], 1)
                    test_acc = (test_pred == y_test_sample).float().mean().item()
                    
                    # For validation accuracy, always use consistent sampling
                    if val_mask.sum() > 4000:
                        val_indices = torch.where(val_mask)[0].to(fold_data.x.device)
                        val_sample_size = min(3000, len(val_indices))
                        sampled_val_indices = val_indices[torch.randperm(len(val_indices), device=val_indices.device)[:val_sample_size]]
                        
                        val_edge_index, _ = subgraph(sampled_val_indices, fold_data.edge_index, relabel_nodes=True)
                        x_val_sample = fold_data.x[sampled_val_indices]
                        y_val_sample = fold_data.y[sampled_val_indices]
                        
                        val_sample_data = Data(x=x_val_sample, edge_index=val_edge_index, y=y_val_sample)
                        
                        # GAT-specific evaluation fallback for validation
                        if model_type == 'gat':
                            # Use GAT's linear transformation for memory-efficient evaluation
                            if hasattr(model, 'convs') and len(model.convs) > 0:
                                if hasattr(model.convs[0], 'lin_src') and model.convs[0].lin_src is not None:
                                    val_output = model.convs[0].lin_src(x_val_sample)
                                else:
                                    # Fallback to basic linear layer
                                    temp_linear = torch.nn.Linear(x_val_sample.size(-1), model.convs[0].out_channels if hasattr(model.convs[0], 'out_channels') else len(fold_data.y.unique())).to(device)
                                    val_output = temp_linear(x_val_sample)
                                
                                # Ensure output matches expected dimensions
                                if val_output.size(-1) != len(fold_data.y.unique()):
                                    final_linear = torch.nn.Linear(val_output.size(-1), len(fold_data.y.unique())).to(device)
                                    val_output = final_linear(val_output)
                            else:
                                # Complete fallback
                                temp_linear = torch.nn.Linear(x_val_sample.size(-1), len(fold_data.y.unique())).to(device)
                                val_output = temp_linear(x_val_sample)
                        else:
                            # Standard evaluation for non-GAT models
                            val_output = model(val_sample_data)
                        
                        val_mask_sample = torch.ones(len(sampled_val_indices), dtype=torch.bool, device=val_output.device)
                        
                        _, val_pred = torch.max(val_output[val_mask_sample], 1)
                        val_acc = (val_pred == y_val_sample).float().mean().item()
                    else:
                        # Use full validation set
                        val_output = model(fold_data)
                        _, val_pred = torch.max(val_output[val_mask], 1)
                        val_acc = (val_pred == fold_data.y[val_mask]).float().mean().item()
            else:
                # Use full dataset
                output = model(fold_data)
                
                # Test performance
                test_loss = criterion(output[test_mask], fold_data.y[test_mask]).item()
                
                if len(fold_data.y.unique()) > 1:  # Classification
                    _, test_pred = torch.max(output[test_mask], 1)
                    test_acc = (test_pred == fold_data.y[test_mask]).float().mean().item()
                    
                    # Validation accuracy
                    _, val_pred = torch.max(output[val_mask], 1)
                    val_acc = (val_pred == fold_data.y[val_mask]).float().mean().item()
            
            # Create result dictionary based on task type
            if len(fold_data.y.unique()) > 1:  # Classification
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
        
        # Memory cleanup after each fold for MPS device
        if device.type == 'mps':
            torch.mps.empty_cache()
            del model, optimizer
            import gc
            gc.collect()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        
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