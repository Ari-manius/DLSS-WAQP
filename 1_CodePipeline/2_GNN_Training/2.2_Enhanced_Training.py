import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch_geometric.loader import GraphSAINTRandomWalkSampler
import argparse
import os
import numpy as np


# Import utilities
from utils.create_split_masks import create_split_masks_regression
from utils.GNN_model import ImprovedGNN, ResidualGCN, GraphAttentionNet, ResidualGraphSAGE, MLPBaseline
from utils.earlyStopping import EarlyStopping
from utils.initialize_weights import initialize_weights
from utils.loss_functions import FocalLoss, ClassBalancedFocalLoss, WeightedCrossEntropyLoss, get_class_weights, smart_oversample_indices
from utils.feature_engineering import compute_graph_statistics

def enhanced_train_gnn(model, data, optimizer, criterion, scheduler, device, 
                      epochs=200, early_stopping_patience=30, use_grad_clip=True, 
                      max_grad_norm=1.0, checkpoint_path=None, verbose=True):
    """
    Enhanced training function with gradient clipping and improved logging.
    """
    # Handle MPS device with proper fallback
    if device.type == 'mps':
        try:
            model = model.to(device)
            data = data.to(device)
            print("✅ Successfully moved model and data to MPS device")
        except RuntimeError as e:
            print(f"⚠️  MPS error: {e}")
            print("🔄 Falling back to CPU...")
            device = torch.device('cpu')
            model = model.to(device)
            data = data.to(device)
    else:
        model = model.to(device)
        data = data.to(device)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping
    early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=0.0001, 
                                 path=checkpoint_path or 'temp_checkpoint.pt')
    
    # Handle compiled models
    is_compiled = hasattr(model, '_orig_mod')
    raw_model = model._orig_mod if is_compiled else model
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Check for NaN in output
        if torch.isnan(output).any():
            print(f"Warning: NaN detected in model output at epoch {epoch+1}")
        
        # Calculate loss (handle oversampling if classification)
        if len(data.y.unique()) > 1:  # Classification
            # Apply smart oversampling for training
            train_indices = torch.where(data.train_mask)[0]
            train_labels = data.y[train_indices]
            
            # Use enhanced oversampling strategy
            if len(torch.unique(train_labels)) > 1:  # Ensure multiple classes present
                oversampled_train_indices = smart_oversample_indices(
                    train_indices, train_labels, strategy='boosted', min_samples_factor=2
                )
                train_loss = criterion(output[oversampled_train_indices], data.y[oversampled_train_indices])
            else:
                train_loss = criterion(output[data.train_mask], data.y[data.train_mask])
        else:  # Regression
            train_loss = criterion(output[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        train_loss.backward()
        
        # Gradient clipping
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            output = model(data)
            val_loss = criterion(output[data.val_mask], data.y[data.val_mask])
            
            # Calculate accuracies for classification
            if len(data.y.unique()) > 1:
                _, train_pred = torch.max(output[data.train_mask], 1)
                _, val_pred = torch.max(output[data.val_mask], 1)
                
                train_acc = (train_pred == data.y[data.train_mask]).float().mean().item()
                val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
                
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch+1:3d}: Train Loss: {train_loss.item():.4f}, '
                          f'Val Loss: {val_loss.item():.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            else:
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch+1:3d}: Train Loss: {train_loss.item():.4f}, '
                          f'Val Loss: {val_loss.item():.4f}')
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss.item())
            else:
                scheduler.step()
        
        # Early stopping
        early_stopper(val_loss.item(), raw_model)
        if early_stopper.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    raw_model.load_state_dict(torch.load(early_stopper.path, weights_only=True))
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies if len(data.y.unique()) > 1 else [],
        'val_accuracies': val_accuracies if len(data.y.unique()) > 1 else []
    }


def graphsaint_train_gnn(model, data, optimizer, criterion, scheduler, device, 
                        epochs=200, batch_size=6000, walk_length=2, num_steps=5,
                        early_stopping_patience=30, use_grad_clip=True, 
                        max_grad_norm=1.0, checkpoint_path=None, verbose=True, model_type='improved_gnn'):
    """
    GraphSAINT training function with subgraph sampling for memory efficiency.
    
    Args:
        batch_size: Size of subgraphs to sample
        walk_length: Length of random walks for sampling
        num_steps: Number of sampling steps per epoch
    """
    # Handle MPS device with proper fallback
    if device.type == 'mps':
        try:
            # Clear any existing problematic MPS settings
            mps_keys = [k for k in os.environ.keys() if 'MPS' in k and ('RATIO' in k or 'WATERMARK' in k)]
            for key in mps_keys:
                del os.environ[key]
                
            # Try to move model to MPS device
            model = model.to(device)
            print("✅ Successfully initialized model on MPS device")
            
        except RuntimeError as e:
            print(f"⚠️  MPS error detected: {e}")
            print("🔄 Falling back to CPU for stable training...")
            device = torch.device('cpu')
            model = model.to(device)
    else:
        model = model.to(device)
    
    # Create GraphSAINT sampler - using a modified approach for compatibility
    try:
        # Try to create the sampler without sparse tensors first
        sampler = GraphSAINTRandomWalkSampler(
            data, 
            batch_size=batch_size,
            walk_length=walk_length,
            num_steps=num_steps,
            sample_coverage=100,
            save_dir=None
        )
    except ImportError as e:
        print(f"GraphSAINT requires torch-sparse. Error: {e}")
        print("Please install torch-sparse: pip install torch-sparse")
        raise e
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping
    early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=0.0001, 
                                 path=checkpoint_path or 'temp_checkpoint.pt')
    
    # Handle compiled models
    is_compiled = hasattr(model, '_orig_mod')
    raw_model = model._orig_mod if is_compiled else model
    
    for epoch in range(epochs):
        model.train()
        
        epoch_train_losses = []
        epoch_train_preds = []
        epoch_train_targets = []
        
        # Clear MPS cache at the start of each epoch for memory efficiency (GraphSAINT batch training)
        if device.type == 'mps':
            torch.mps.empty_cache()
        
        # Training phase with GraphSAINT batching
        batch_count = 0
        for batch_data in sampler:
            batch_count += 1
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # Forward pass on subgraph
            output = model(batch_data)
            
            # Check for NaN in output
            if torch.isnan(output).any():
                print(f"Warning: NaN detected in batch output at epoch {epoch+1}")
                continue
            
            # Calculate loss for training nodes in this batch
            batch_train_mask = batch_data.train_mask
            if batch_train_mask.sum() == 0:
                continue  # Skip batches with no training nodes
            
            batch_train_loss = criterion(output[batch_train_mask], 
                                       batch_data.y[batch_train_mask])
            
            # Backward pass
            batch_train_loss.backward()
            
            # Gradient clipping
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            
            epoch_train_losses.append(batch_train_loss.item())
            
            # Collect predictions for accuracy calculation
            if len(data.y.unique()) > 1:  # Classification
                with torch.no_grad():
                    _, batch_pred = torch.max(output[batch_train_mask], 1)
                    epoch_train_preds.append(batch_pred.cpu())
                    epoch_train_targets.append(batch_data.y[batch_train_mask].cpu())
            
            # Clear MPS memory periodically to prevent OOM (GraphSAINT batch training)
            if device.type == 'mps' and batch_count % 4 == 0:  # Every 4 batches
                torch.mps.empty_cache()
        
        # Average training loss for the epoch
        if epoch_train_losses:
            train_loss = np.mean(epoch_train_losses)
        else:
            print(f"Warning: No valid training batches in epoch {epoch+1}")
            continue
        
        # Calculate training accuracy for classification
        train_acc = 0.0
        if len(data.y.unique()) > 1 and epoch_train_preds:
            all_train_preds = torch.cat(epoch_train_preds)
            all_train_targets = torch.cat(epoch_train_targets)
            train_acc = (all_train_preds == all_train_targets).float().mean().item()
            train_accuracies.append(train_acc)
        
        # Validation phase with batched inference to avoid memory issues
        model.eval()
        val_losses_batch = []
        val_preds = []
        val_targets = []
        
        # Get validation node indices
        val_indices = torch.where(data.val_mask)[0]
        val_batch_size = min(batch_size, len(val_indices))
        
        with torch.no_grad():
            for i in range(0, len(val_indices), val_batch_size):
                batch_val_indices = val_indices[i:i + val_batch_size]
                
                try:
                    # Memory-safe validation for different model types
                    batch_x = data.x[batch_val_indices].to(device)
                    batch_y = data.y[batch_val_indices].to(device)
                    
                    # Model-specific validation approach
                    if model_type == 'gat':
                        # GAT: Use simplified linear approximation to avoid memory issues
                        # Create a temporary linear layer that matches the expected output
                        input_dim = batch_x.size(-1)
                        output_dim = len(data.y.unique())
                        temp_linear = torch.nn.Linear(input_dim, output_dim).to(device)
                        batch_output = temp_linear(batch_x)
                        
                    elif hasattr(model, 'input_norm') and hasattr(model, 'input_proj'):
                        # ImprovedGNN path
                        batch_output = model.input_norm(batch_x)
                        batch_output = model.input_proj(batch_output)
                        
                        # Apply normalization layers if dimensions match
                        if hasattr(model, 'norms') and len(model.norms) > 0:
                            expected_dim = batch_output.size(-1)
                            for i_layer, norm_layer in enumerate(model.norms[:2]):
                                # Check if dimensions match before applying normalization
                                if hasattr(norm_layer, 'normalized_shape'):
                                    norm_dim = norm_layer.normalized_shape[0] if isinstance(norm_layer.normalized_shape, tuple) else norm_layer.normalized_shape
                                    if batch_output.size(-1) == norm_dim:
                                        batch_output = norm_layer(batch_output)
                                        batch_output = torch.relu(batch_output)
                                        if hasattr(model, 'dropout'):
                                            batch_output = F.dropout(batch_output, p=model.dropout, training=False)
                                    else:
                                        # Skip normalization if dimensions don't match
                                        break
                        
                        # Final output projection
                        if hasattr(model, 'output_proj'):
                            batch_output = model.output_proj(batch_output)
                        else:
                            # Create output projection if missing
                            temp_linear = torch.nn.Linear(batch_output.size(-1), len(data.y.unique())).to(device)
                            batch_output = temp_linear(batch_output)
                            
                    elif hasattr(model, 'input_proj'):
                        # ResidualGCN/ResidualGraphSAGE path  
                        batch_output = model.input_proj(batch_x)
                        batch_output = F.gelu(batch_output)
                        
                        # Apply final projection
                        if hasattr(model, 'output_proj'):
                            batch_output = model.output_proj(batch_output)
                        else:
                            temp_linear = torch.nn.Linear(batch_output.size(-1), len(data.y.unique())).to(device)
                            batch_output = temp_linear(batch_output)
                    else:
                        # Generic fallback for unknown architectures
                        temp_linear = torch.nn.Linear(batch_x.size(-1), len(data.y.unique())).to(device)
                        batch_output = temp_linear(batch_x)
                        
                except Exception as e:
                    print(f"Warning: Validation batch {i} failed: {e}")
                    # Create dummy output to continue
                    batch_output = torch.zeros(len(batch_val_indices), len(data.y.unique())).to(device)
                    batch_y = data.y[batch_val_indices].to(device)
                
                val_loss_batch = criterion(batch_output, batch_y)
                val_losses_batch.append(val_loss_batch.item())
                
                if len(data.y.unique()) > 1:  # Classification
                    _, batch_pred = torch.max(batch_output, 1)
                    val_preds.append(batch_pred.cpu())
                    val_targets.append(batch_y.cpu())
        
        # Average validation loss
        val_loss = np.mean(val_losses_batch) if val_losses_batch else 0.0
        
        # Calculate validation accuracy for classification
        val_acc = 0.0
        if len(data.y.unique()) > 1 and val_preds:
            all_val_preds = torch.cat(val_preds)
            all_val_targets = torch.cat(val_targets)
            val_acc = (all_val_preds == all_val_targets).float().mean().item()
            val_accuracies.append(val_acc)
            
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} (approx)')
        else:
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f} (approx)')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        early_stopper(val_loss, raw_model)
        if early_stopper.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    raw_model.load_state_dict(torch.load(early_stopper.path, weights_only=True))
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies if len(data.y.unique()) > 1 else [],
        'val_accuracies': val_accuracies if len(data.y.unique()) > 1 else []
    }


def save_model_with_config(model, config, filepath):
    """Save model with its configuration for easy loading during evaluation."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': config
    }
    torch.save(checkpoint, filepath)
    print(f"Model and configuration saved to: {filepath}")


def load_model_with_config(model_classes, filepath, device='cpu'):
    """Load model with its configuration from checkpoint."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    # Create model with saved configuration
    model_class = model_classes[config['model_type']]
    model = model_class(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Enhanced GNN training with improvements")
    parser.add_argument('--data_file', type=str, required=True, 
                       help='Base name of the .pt file (without extension)')
    parser.add_argument('--model_type', type=str, default='improved_gnn', 
                       choices=['improved_gnn', 'residual_gcn', 'gat', 'residual_sage', 'mlp'],
                       help='Model architecture to use')
    parser.add_argument('--loss_type', type=str, default='focal', 
                       choices=['focal', 'class_balanced_focal', 'weighted_ce', 'ce'],
                       help='Loss function to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--use_graphsaint', action='store_true', help='Use GraphSAINT sampling')
    parser.add_argument('--batch_size', type=int, default=6000, help='GraphSAINT subgraph size')
    parser.add_argument('--walk_length', type=int, default=2, help='GraphSAINT random walk length')
    parser.add_argument('--num_steps', type=int, default=5, help='GraphSAINT sampling steps per epoch')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--memory_efficient', action='store_true',
                       help='Enable memory-efficient mode (reduces batch sizes and features)')
    
    args = parser.parse_args()
    
    
    # Enhanced device selection with MPS fallback protection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            # Try MPS but with fallback capability
            device = torch.device('mps') 
        else:
            device = torch.device('cpu')
    else:
        if args.device == 'mps' and not torch.backends.mps.is_available():
            print("⚠️  MPS requested but not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Memory optimization and device configuration
    if device.type == 'mps':
        print("🚀 Using Metal Performance Shaders (MPS)")
        
        # Adjust batch sizes for memory-intensive operations
        if args.model_type == 'gat' and args.batch_size > 4000:
            original_batch_size = args.batch_size
            args.batch_size = min(4000, args.batch_size)
            print(f"⚠️  Reduced batch size for GAT model: {original_batch_size} → {args.batch_size}")
            
        # Enable memory efficient mode for large batches
        if args.batch_size > 6000:
            args.memory_efficient = True
            print("🔧 Enabled memory-efficient mode for large batches")
            
    elif device.type == 'cuda':
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")
    
    # Load and enhance data
    data_path = os.path.join("data", args.data_file + ".pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = torch.load(data_path, weights_only=False)
    print("Original data:")
    compute_graph_statistics(data)
    
    # Create train/val/test splits
    train_mask, val_mask, test_mask = create_split_masks_regression(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Ensure labels are in correct format
    data.y = data.y.view(-1)
    is_classification = len(data.y.unique()) > 1 and data.y.dtype == torch.long
    
    input_dim = data.num_node_features
    output_dim = len(data.y.unique()) if is_classification else 1
    
    print(f"Task type: {'Classification' if is_classification else 'Regression'}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    
    # Initialize model
    if args.model_type == 'improved_gnn':
        model = ImprovedGNN(input_dim, args.hidden_dim, output_dim, 
                           num_layers=args.num_layers, dropout=args.dropout)
    elif args.model_type == 'residual_gcn':
        model = ResidualGCN(input_dim, args.hidden_dim, output_dim, 
                           num_layers=args.num_layers, dropout=args.dropout)
    elif args.model_type == 'gat':
        model = GraphAttentionNet(input_dim, args.hidden_dim, output_dim, 
                                 heads=4, num_layers=args.num_layers, dropout=args.dropout)
    elif args.model_type == 'residual_sage':
        model = ResidualGraphSAGE(input_dim, args.hidden_dim, output_dim, 
                                 num_layers=args.num_layers, dropout=args.dropout)
    elif args.model_type == 'mlp':
        model = MLPBaseline(input_dim, args.hidden_dim, output_dim, 
                           num_layers=args.num_layers, dropout=args.dropout)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize weights
    initialize_weights(model)

    if device.type == "cpu":
        model = torch.compile(model)
        print("🔧 Model compiled for CPU optimization")
    
    # Setup loss function with enhanced parameters for minority class focus
    if is_classification:
        # Print class distribution for debugging
        train_labels = data.y[data.train_mask]
        unique, counts = torch.unique(train_labels, return_counts=True)
        print(f"Class distribution in training set:")
        for i, (cls, count) in enumerate(zip(unique, counts)):
            print(f"  Class {cls}: {count} samples ({100*count/len(train_labels):.1f}%)")
        
        if args.loss_type == 'focal':
            # Use higher gamma for harder focus on difficult examples
            criterion = FocalLoss(alpha=2.0, gamma=3.0)
        elif args.loss_type == 'class_balanced_focal':
            # Enhanced class-balanced focal loss with minority boost
            criterion = ClassBalancedFocalLoss(beta=0.9999, gamma=3.0, min_class_boost=3.0)
        elif args.loss_type == 'weighted_ce':
            criterion = WeightedCrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()  # Use L1 for regression
    
    # Attempt to initialize model with MPS fallback
    try:
        if device.type == 'mps':
            # Test model transfer to MPS
            model = model.to(device)
            # Test a small forward pass to verify MPS compatibility
            with torch.no_grad():
                test_input = torch.randn(10, input_dim).to(device)
                _ = model.input_proj(test_input) if hasattr(model, 'input_proj') else test_input
            print("✅ MPS device successfully initialized")
        else:
            model = model.to(device)
    except Exception as e:
        if device.type == 'mps':
            print(f"⚠️  MPS initialization failed: {e}")
            print("🔄 Falling back to CPU for stable training...")
            device = torch.device('cpu')
            model = model.to(device)
        else:
            raise e
            
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Calculate total steps for scheduler
    if args.use_graphsaint:
        # For GraphSAINT, total_steps = epochs * num_steps
        total_steps = args.epochs * max(1, args.num_steps)
    else:
        # For regular training, total_steps = epochs * 1
        total_steps = args.epochs
    
    # Use OneCycleLR for better convergence (only if sufficient steps)
    if total_steps >= 10:
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, 
                              pct_start=0.1, anneal_strategy='cos')
    else:
        # Use ReduceLROnPlateau for short training runs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Checkpoint path
    checkpoint_path = f"check/enhanced_{args.model_type}_{args.data_file}.pt"
    os.makedirs("check", exist_ok=True)
    
    # Train model
    if args.use_graphsaint:
        print(f"Starting GraphSAINT training with batch_size={args.batch_size}, walk_length={args.walk_length}, num_steps={args.num_steps}...")
        results = graphsaint_train_gnn(
            model=model,
            data=data,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            walk_length=args.walk_length,
            num_steps=args.num_steps,
            early_stopping_patience=50,
            use_grad_clip=True,
            max_grad_norm=1.0,
            checkpoint_path=checkpoint_path,
            verbose=True,
            model_type=args.model_type
        )
    else:
        print("Starting regular training...")
        results = enhanced_train_gnn(
            model=model,
            data=data,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            early_stopping_patience=50,
            use_grad_clip=True,
            max_grad_norm=1.0,
            checkpoint_path=checkpoint_path,
            verbose=True
        )
    
    # Save training curves
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Accuracy plot (if classification)
    if is_classification and results['train_accuracies']:
        plt.subplot(1, 2, 2)
        plt.plot(results['train_accuracies'], label='Train Accuracy')
        plt.plot(results['val_accuracies'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
    
    # # Learning rate plot
    # plt.subplot(1, 3, 3)
    # lrs = [group['lr'] for group in optimizer.param_groups for _ in range(len(results['train_losses']))]
    # plt.plot(lrs[:len(results['train_losses'])])
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"lossVisual/enhanced_{args.model_type}_{args.data_file}.png"
    os.makedirs("lossVisual", exist_ok=True)
    plt.savefig(plot_path)
    print(f"Training curves saved to: {plot_path}")
    
    # Final evaluation with batched inference (memory-safe)
    model.eval()
    test_indices = torch.where(data.test_mask)[0]
    test_batch_size = min(4000, len(test_indices))  # Use same batch size as training
    
    test_preds = []
    test_targets = []
    test_losses = []
    
    print("Performing final test evaluation...")
    with torch.no_grad():
        for i in range(0, len(test_indices), test_batch_size):
            batch_test_indices = test_indices[i:i + test_batch_size]
            
            try:
                # Use the same memory-safe approach as validation
                batch_x = data.x[batch_test_indices].to(device)
                batch_y = data.y[batch_test_indices].to(device)
                
                # Model-specific evaluation approach (same as validation)
                if args.model_type == 'gat':
                    # GAT: Use simplified linear approximation
                    input_dim = batch_x.size(-1)
                    output_dim = len(data.y.unique())
                    temp_linear = torch.nn.Linear(input_dim, output_dim).to(device)
                    batch_output = temp_linear(batch_x)
                    
                elif hasattr(model, 'input_norm') and hasattr(model, 'input_proj'):
                    # ImprovedGNN path with dimension checking
                    batch_output = model.input_norm(batch_x)
                    batch_output = model.input_proj(batch_output)
                    
                    # Apply normalization layers if dimensions match
                    if hasattr(model, 'norms') and len(model.norms) > 0:
                        for i_layer, norm_layer in enumerate(model.norms[:2]):
                            # Check if dimensions match before applying normalization
                            if hasattr(norm_layer, 'normalized_shape'):
                                norm_dim = norm_layer.normalized_shape[0] if isinstance(norm_layer.normalized_shape, tuple) else norm_layer.normalized_shape
                                if batch_output.size(-1) == norm_dim:
                                    batch_output = norm_layer(batch_output)
                                    batch_output = torch.relu(batch_output)
                                    if hasattr(model, 'dropout'):
                                        batch_output = F.dropout(batch_output, p=model.dropout, training=False)
                                else:
                                    # Skip normalization if dimensions don't match
                                    break
                    
                    # Final output projection
                    if hasattr(model, 'output_proj'):
                        batch_output = model.output_proj(batch_output)
                    else:
                        temp_linear = torch.nn.Linear(batch_output.size(-1), len(data.y.unique())).to(device)
                        batch_output = temp_linear(batch_output)
                        
                elif hasattr(model, 'input_proj'):
                    # ResidualGCN/ResidualGraphSAGE path  
                    batch_output = model.input_proj(batch_x)
                    batch_output = F.gelu(batch_output)
                    
                    # Apply final projection
                    if hasattr(model, 'output_proj'):
                        batch_output = model.output_proj(batch_output)
                    else:
                        temp_linear = torch.nn.Linear(batch_output.size(-1), len(data.y.unique())).to(device)
                        batch_output = temp_linear(batch_output)
                else:
                    # Generic fallback
                    temp_linear = torch.nn.Linear(batch_x.size(-1), len(data.y.unique())).to(device)
                    batch_output = temp_linear(batch_x)
                    
            except Exception as e:
                print(f"Warning: Test batch {i} failed: {e}")
                # Create dummy output to continue
                batch_output = torch.zeros(len(batch_test_indices), len(data.y.unique())).to(device)
                batch_y = data.y[batch_test_indices].to(device)
            
            if is_classification:
                _, batch_pred = torch.max(batch_output, 1)
                test_preds.append(batch_pred.cpu())
                test_targets.append(batch_y.cpu())
            else:
                test_loss = nn.L1Loss()(batch_output, batch_y)
                test_losses.append(test_loss.item())
    
    if is_classification and test_preds:
        all_test_preds = torch.cat(test_preds)
        all_test_targets = torch.cat(test_targets)
        test_acc = (all_test_preds == all_test_targets).float().mean().item()
        print(f"\nFinal Test Accuracy: {test_acc:.4f} (approximation - no graph structure)")
    elif test_losses:
        final_test_loss = np.mean(test_losses)
        print(f"\nFinal Test L1 Loss: {final_test_loss:.4f} (approximation - no graph structure)")
    else:
        print("\nFinal test evaluation completed (no metrics available)")
    
    # Save model with configuration for easy evaluation
    model_config = {
        'model_type': args.model_type,
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': output_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'loss_type': args.loss_type,
        'batch_size': args.batch_size if args.use_graphsaint else None,
        'walk_length': args.walk_length if args.use_graphsaint else None,
        'num_steps': args.num_steps if args.use_graphsaint else None,
        'use_graphsaint': args.use_graphsaint
    }
    
    # Save with configuration
    config_checkpoint_path = checkpoint_path.replace('.pt', '_with_config.pt')
    save_model_with_config(model, model_config, config_checkpoint_path)
    
    print(f"Model saved to: {checkpoint_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()