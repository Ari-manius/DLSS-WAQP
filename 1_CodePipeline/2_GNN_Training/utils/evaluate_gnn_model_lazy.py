from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import numpy as np
from .lazy_graph_loader import LazyGraphLoader

def evaluate_gnn_model_lazy(data_path, model, mask_type='test', device='cpu', batch_size=1000):
    """
    Evaluate GNN model using lazy loading to avoid loading the entire graph into memory.
    
    Args:
        data_path: Path to the graph data file
        model: The GNN model to evaluate
        mask_type: 'test' or 'val' for evaluation mask
        device: Device to run evaluation on
        batch_size: Number of nodes to process per batch
    """
    model = model.to(device)
    model.eval()
    
    # Initialize lazy loader
    loader = LazyGraphLoader(data_path, batch_size=batch_size, device=device)
    
    all_predictions = []
    all_targets = []
    
    # Get total number of test nodes for progress tracking
    if mask_type == 'test':
        total_test_nodes = loader.metadata['test_mask'].sum().item()
    else:
        total_test_nodes = loader.metadata['val_mask'].sum().item()
    
    total_batches = (total_test_nodes + batch_size - 1) // batch_size
    processed_nodes = 0
    
    #print(f"Evaluating model on {mask_type} set using lazy loading...")
    #print(f"Total {mask_type} nodes: {total_test_nodes}, Batch size: {batch_size}, Total batches: {total_batches}")
    
    with torch.no_grad():
        for batch_idx, (target_nodes, expanded_nodes) in enumerate(loader.get_test_node_batches(mask_type)):
            processed_nodes += len(target_nodes)
            progress = (processed_nodes / total_test_nodes) * 100
            #print(f"Batch {batch_idx + 1}/{total_batches}: Processing {len(target_nodes)} nodes ({processed_nodes}/{total_test_nodes}, {progress:.1f}%)")
            
            # Load subgraph for this batch
            subgraph_data = loader.get_subgraph_batch(expanded_nodes)
            
            # Get model predictions on subgraph
            outputs = model(subgraph_data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Map back to original target nodes
            # Find indices of target nodes in the subgraph
            target_indices_in_subgraph = []
            for target_node in target_nodes:
                idx = torch.where(expanded_nodes == target_node)[0]
                if len(idx) > 0:
                    target_indices_in_subgraph.append(idx[0])
            
            target_indices_in_subgraph = torch.tensor(target_indices_in_subgraph, device=device)
            
            # Get predictions for target nodes only
            if len(target_indices_in_subgraph) > 0:
                _, predicted = torch.max(outputs[target_indices_in_subgraph], 1)
                targets = subgraph_data.y[target_indices_in_subgraph]
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    y_preds = np.array(all_predictions)
    y_true = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_preds)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(y_true, y_preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_preds))
    
    results = classification_report(y_true, y_preds, digits=3, output_dict=True)
    matrix = confusion_matrix(y_true, y_preds)
    
    return results, matrix

def evaluate_gnn_regression_lazy(data_path, model, mask_type='test', device='cpu', batch_size=1000):
    """
    Evaluate GNN regression model using lazy loading.
    """
    model = model.to(device)
    model.eval()
    
    loader = LazyGraphLoader(data_path, batch_size=batch_size, device=device)
    
    all_predictions = []
    all_targets = []
    
    # Get total number of test nodes for progress tracking
    if mask_type == 'test':
        total_test_nodes = loader.metadata['test_mask'].sum().item()
    else:
        total_test_nodes = loader.metadata['val_mask'].sum().item()
    
    total_batches = (total_test_nodes + batch_size - 1) // batch_size
    processed_nodes = 0
    
    print(f"Evaluating regression model on {mask_type} set using lazy loading...")
    print(f"Total {mask_type} nodes: {total_test_nodes}, Batch size: {batch_size}, Total batches: {total_batches}")
    
    with torch.no_grad():
        for batch_idx, (target_nodes, expanded_nodes) in enumerate(loader.get_test_node_batches(mask_type)):
            processed_nodes += len(target_nodes)
            progress = (processed_nodes / total_test_nodes) * 100
            print(f"Batch {batch_idx + 1}/{total_batches}: Processing {len(target_nodes)} nodes ({processed_nodes}/{total_test_nodes}, {progress:.1f}%)")
            
            subgraph_data = loader.get_subgraph_batch(expanded_nodes)
            outputs = model(subgraph_data)
            
            # Map back to target nodes
            target_indices_in_subgraph = []
            for target_node in target_nodes:
                idx = torch.where(expanded_nodes == target_node)[0]
                if len(idx) > 0:
                    target_indices_in_subgraph.append(idx[0])
            
            target_indices_in_subgraph = torch.tensor(target_indices_in_subgraph, device=device)
            
            if len(target_indices_in_subgraph) > 0:
                predictions = outputs[target_indices_in_subgraph].cpu().numpy().flatten()
                targets = subgraph_data.y[target_indices_in_subgraph].cpu().numpy().flatten()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
    
    # Convert to numpy arrays
    outputs = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Calculate regression metrics
    mse = mean_squared_error(targets, outputs)
    mae = mean_absolute_error(targets, outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, outputs)
    
    epsilon = 1e-8
    mape = np.mean(np.abs((targets - outputs) / (targets + epsilon))) * 100
    mpe = np.mean((targets - outputs) / (targets + epsilon)) * 100
    
    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²:    {r2:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    print(f"MPE:   {mpe:.2f}%")
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "MPE": mpe
    }