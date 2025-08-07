import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import argparse
import os
import numpy as np

# Import utilities
from utils.create_split_masks import create_split_masks_regression
from utils.GNN_model import ImprovedGNN, ResidualGCN, GraphAttentionNet, ResidualGraphSAGE
from utils.earlyStopping import EarlyStopping
from utils.initialize_weights import initialize_weights
from utils.loss_functions import FocalLoss, ClassBalancedFocalLoss, WeightedCrossEntropyLoss, get_class_weights
from utils.feature_engineering import compute_graph_statistics

def enhanced_train_gnn(model, data, optimizer, criterion, scheduler, device, 
                      epochs=200, early_stopping_patience=30, use_grad_clip=True, 
                      max_grad_norm=1.0, checkpoint_path=None, verbose=True):
    """
    Enhanced training function with gradient clipping and improved logging.
    """
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
            # Apply oversampling for training
            train_indices = torch.where(data.train_mask)[0]
            train_labels = data.y[train_indices]
            
            # Simple oversampling - repeat minority class samples
            class_counts = torch.bincount(train_labels)
            if class_counts.min() > 0:  # Ensure all classes are present
                max_count = class_counts.max()
                oversampled_indices = []
                
                for class_idx in range(len(class_counts)):
                    class_mask = train_labels == class_idx
                    class_indices = train_indices[class_mask]
                    
                    if len(class_indices) == 0:
                        continue
                    
                    # Repeat to match max count
                    n_repeats = max_count // len(class_indices)
                    remainder = max_count % len(class_indices)
                    
                    repeated = class_indices.repeat(n_repeats)
                    if remainder > 0:
                        perm_indices = torch.randperm(len(class_indices))[:remainder]
                        repeated = torch.cat([repeated, class_indices[perm_indices]])
                    oversampled_indices.append(repeated)
                
                oversampled_train_indices = torch.cat(oversampled_indices)
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


def main():
    parser = argparse.ArgumentParser(description="Enhanced GNN training with improvements")
    parser.add_argument('--data_file', type=str, required=True, 
                       help='Base name of the .pt file (without extension)')
    parser.add_argument('--model_type', type=str, default='improved_gnn', 
                       choices=['improved_gnn', 'residual_gcn', 'gat', 'residual_sage'],
                       help='Model architecture to use')
    parser.add_argument('--loss_type', type=str, default='focal', 
                       choices=['focal', 'class_balanced_focal', 'weighted_ce', 'ce'],
                       help='Loss function to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    
    args = parser.parse_args()
    
    device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
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
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize weights
    initialize_weights(model)
    model = torch.compile(model)
    
    # Setup loss function
    if is_classification:
        if args.loss_type == 'focal':
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif args.loss_type == 'class_balanced_focal':
            criterion = ClassBalancedFocalLoss(beta=0.9999, gamma=2.0)
        elif args.loss_type == 'weighted_ce':
            criterion = WeightedCrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()  # Use L1 for regression
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Use OneCycleLR for better convergence
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, 
                          steps_per_epoch=1, pct_start=0.1, anneal_strategy='cos')
    
    # Checkpoint path
    checkpoint_path = f"check/enhanced_{args.model_type}_{args.data_file}.pt"
    os.makedirs("check", exist_ok=True)
    
    # Train model
    print("Starting training...")
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
    plt.subplot(1, 3, 1)
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Accuracy plot (if classification)
    if is_classification and results['train_accuracies']:
        plt.subplot(1, 3, 2)
        plt.plot(results['train_accuracies'], label='Train Accuracy')
        plt.plot(results['val_accuracies'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    lrs = [group['lr'] for group in optimizer.param_groups for _ in range(len(results['train_losses']))]
    plt.plot(lrs[:len(results['train_losses'])])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"lossVisual/enhanced_{args.model_type}_{args.data_file}.png"
    os.makedirs("lossVisual", exist_ok=True)
    plt.savefig(plot_path)
    print(f"Training curves saved to: {plot_path}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        output = model(data.to(device))
        
        if is_classification:
            _, test_pred = torch.max(output[data.test_mask], 1)
            test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()
            print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        else:
            test_loss = nn.L1Loss()(output[data.test_mask], data.y[data.test_mask]).item()
            print(f"\nFinal Test L1 Loss: {test_loss:.4f}")
    
    print(f"Model saved to: {checkpoint_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()