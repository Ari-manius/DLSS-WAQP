#!/usr/bin/env python3
"""
Cross-Validation Training Module
Modified version of Enhanced_Training.py that supports cross-validation splits.
"""

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
import json
import pickle
from datetime import datetime

# Import utilities
from utils.create_split_masks import create_split_masks_regression
from utils.GNN_model import ImprovedGNN, ResidualGCN, GraphAttentionNet, ResidualGraphSAGE, MLPBaseline
from utils.earlyStopping import EarlyStopping
from utils.initialize_weights import initialize_weights
from utils.loss_functions import FocalLoss, ClassBalancedFocalLoss, WeightedCrossEntropyLoss, get_class_weights, smart_oversample_indices
from utils.feature_engineering import compute_graph_statistics
from utils.load_graph_data import load_graph_data

def load_cv_splits(splits_file):
    """Load cross-validation splits from file."""
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    return splits

def apply_cv_split(data, fold_data):
    """Apply CV split to data object."""
    train_indices = torch.tensor(fold_data['train_indices'], dtype=torch.long)
    val_indices = torch.tensor(fold_data['val_indices'], dtype=torch.long)
    
    # Create new masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # Use val as test for CV
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[val_indices] = True  # Test on validation set for CV
    
    # Update data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data

def enhanced_train_gnn_cv(model, data, optimizer, criterion, scheduler, device, 
                         epochs=200, early_stopping_patience=30, use_grad_clip=True, 
                         max_grad_norm=1.0, checkpoint_path=None, verbose=True,
                         oversample_strategy='boosted', min_samples_factor=2):
    """Enhanced training function for cross-validation."""
    # Handle MPS device with proper fallback
    if device.type == 'mps':
        try:
            model = model.to(device)
            data = data.to(device)
            if verbose:
                print("✅ Successfully moved model and data to MPS device")
        except RuntimeError as e:
            if verbose:
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
    
    best_val_acc = 0
    best_predictions = None
    best_true_labels = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Check for NaN and infinity in output
        if torch.isnan(output).any():
            if verbose:
                print(f"Warning: NaN detected in model output at epoch {epoch+1}")
        if torch.isinf(output).any():
            if verbose:
                print(f"Warning: Infinity detected in model output at epoch {epoch+1}")
            continue
        
        # Calculate loss (handle oversampling if classification)
        if len(data.y.unique()) > 1:  # Classification
            train_indices = torch.where(data.train_mask)[0].to(data.x.device)
            train_labels = data.y[train_indices]
            
            if len(torch.unique(train_labels)) > 1:  # Ensure multiple classes present
                oversampled_train_indices = smart_oversample_indices(
                    train_indices, train_labels, strategy=oversample_strategy, min_samples_factor=min_samples_factor,
                    max_total_samples=20000
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
            val_output = model(data)
            val_loss = criterion(val_output[data.val_mask], data.y[data.val_mask])
            
            # Calculate accuracies
            if len(data.y.unique()) > 1:  # Classification
                train_pred = output[data.train_mask].argmax(dim=1)
                train_acc = (train_pred == data.y[data.train_mask]).float().mean()
                
                val_pred = val_output[data.val_mask].argmax(dim=1)
                val_acc = (val_pred == data.y[data.val_mask]).float().mean()
                
                # Save best predictions for ensemble
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_predictions = val_pred.cpu().numpy()
                    best_true_labels = data.y[data.val_mask].cpu().numpy()
                
            else:  # Regression
                # For regression, use negative MSE as "accuracy"
                train_mse = F.mse_loss(output[data.train_mask], data.y[data.train_mask])
                val_mse = F.mse_loss(val_output[data.val_mask], data.y[data.val_mask])
                train_acc = -train_mse
                val_acc = -val_mse
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_predictions = val_output[data.val_mask].cpu().numpy()
                    best_true_labels = data.y[data.val_mask].cpu().numpy()
        
        # Record metrics
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_acc.item())
        val_accuracies.append(val_acc.item())
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            else:
                scheduler.step(val_loss)
        
        # Early stopping
        early_stopper(val_loss, raw_model)
        if early_stopper.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Progress reporting
        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    # Load best model
    if os.path.exists(early_stopper.path):
        raw_model.load_state_dict(torch.load(early_stopper.path, map_location=device))
        os.remove(early_stopper.path)  # Clean up temp file
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'predictions': best_predictions,
        'true_labels': best_true_labels
    }

def main():
    parser = argparse.ArgumentParser(description='Cross-Validation GNN Training')
    
    # Data and model arguments
    parser.add_argument('--data_file', type=str, required=True, help='Name of the data file')
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['improved_gnn', 'residual_gcn', 'gat', 'residual_sage', 'mlp'])
    parser.add_argument('--cv_splits_file', type=str, required=True, help='Path to CV splits file')
    parser.add_argument('--fold', type=int, required=True, help='Fold number to train on')
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    
    # Loss function arguments
    parser.add_argument('--loss_type', type=str, default='weighted_ce',
                       choices=['weighted_ce', 'focal', 'class_balanced_focal'])
    parser.add_argument('--cb_focal_beta', type=float, default=0.9999)
    parser.add_argument('--cb_focal_gamma', type=float, default=2.0)
    parser.add_argument('--min_class_boost', type=float, default=1.0)
    parser.add_argument('--oversample_strategy', type=str, default='boosted')
    parser.add_argument('--min_samples_factor', type=int, default=2)
    
    # GAT specific
    parser.add_argument('--heads', type=int, default=4)
    
    # Training arguments
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--use_graphsaint', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=8)
    
    args = parser.parse_args()
    
    print(f"🔄 Training {args.model_type} on fold {args.fold}")
    print("=" * 60)
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"📱 Using device: {device}")
    
    # Load data
    print(f"📂 Loading data: {args.data_file}")
    data = load_graph_data(args.data_file)
    if data is None:
        print("❌ Failed to load data")
        return
    
    # Load CV splits and apply fold
    print(f"📊 Loading CV splits from {args.cv_splits_file}")
    splits = load_cv_splits(args.cv_splits_file)
    fold_data = splits[args.fold]
    data = apply_cv_split(data, fold_data)
    
    print(f"📊 Fold {args.fold} split - Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}")
    
    # Initialize model
    print(f"🧠 Initializing {args.model_type} model")
    
    if args.model_type == 'improved_gnn':
        model = ImprovedGNN(data.x.shape[1], args.hidden_dim, data.y.max().item() + 1,
                           num_layers=args.num_layers, dropout=args.dropout)
    elif args.model_type == 'residual_gcn':
        model = ResidualGCN(data.x.shape[1], args.hidden_dim, data.y.max().item() + 1,
                           num_layers=args.num_layers, dropout=args.dropout)
    elif args.model_type == 'gat':
        model = GraphAttentionNet(data.x.shape[1], args.hidden_dim, data.y.max().item() + 1,
                                 num_layers=args.num_layers, heads=args.heads, dropout=args.dropout)
    elif args.model_type == 'residual_sage':
        model = ResidualGraphSAGE(data.x.shape[1], args.hidden_dim, data.y.max().item() + 1,
                                 num_layers=args.num_layers, dropout=args.dropout)
    elif args.model_type == 'mlp':
        model = MLPBaseline(data.x.shape[1], args.hidden_dim, data.y.max().item() + 1,
                           num_layers=args.num_layers, dropout=args.dropout)
    
    # Initialize weights
    model.apply(initialize_weights)
    
    # Setup loss function
    train_labels = data.y[data.train_mask]
    class_weights = get_class_weights(train_labels)
    
    if args.loss_type == 'weighted_ce':
        criterion = WeightedCrossEntropyLoss(class_weights)
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights)
    elif args.loss_type == 'class_balanced_focal':
        criterion = ClassBalancedFocalLoss(beta=args.cb_focal_beta, gamma=args.cb_focal_gamma,
                                          min_class_boost=args.min_class_boost)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                          steps_per_epoch=1, pct_start=0.3)
    
    # Train model
    checkpoint_path = f"cv_checkpoint_{args.model_type}_fold_{args.fold}.pt"
    
    results = enhanced_train_gnn_cv(
        model, data, optimizer, criterion, scheduler, device,
        epochs=args.epochs, checkpoint_path=checkpoint_path,
        oversample_strategy=args.oversample_strategy,
        min_samples_factor=args.min_samples_factor
    )
    
    # Test on validation set (which serves as test set in CV)
    model.eval()
    with torch.no_grad():
        test_output = model(data)
        test_pred = test_output[data.test_mask].argmax(dim=1)
        test_acc = (test_pred == data.y[data.test_mask]).float().mean()
    
    print(f"📊 Final Results:")
    print(f"  Best Val Acc: {results['best_val_acc']:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    
    # Save results
    os.makedirs("cv_results", exist_ok=True)
    
    # Save numerical results
    fold_results = {
        'model_type': args.model_type,
        'fold': args.fold,
        'test_accuracy': test_acc.item(),
        'best_val_accuracy': results['best_val_acc'],
        'hyperparameters': vars(args),
        'training_history': {
            'train_losses': results['train_losses'],
            'val_losses': results['val_losses'],
            'train_accuracies': results['train_accuracies'],
            'val_accuracies': results['val_accuracies']
        }
    }
    
    results_file = f"cv_results/{args.model_type}_{args.data_file}_fold_{args.fold}_results.json"
    with open(results_file, 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    # Save predictions for ensemble
    pred_data = {
        'predictions': results['predictions'],
        'true_labels': results['true_labels']
    }
    
    pred_file = f"cv_results/{args.model_type}_{args.data_file}_fold_{args.fold}_predictions.pkl"
    with open(pred_file, 'wb') as f:
        pickle.dump(pred_data, f)
    
    # Save model checkpoint
    model_file = f"cv_results/{args.model_type}_{args.data_file}_fold_{args.fold}_model.pt"
    torch.save(model.state_dict(), model_file)
    
    print(f"✅ Results saved:")
    print(f"  Metrics: {results_file}")
    print(f"  Predictions: {pred_file}")
    print(f"  Model: {model_file}")

if __name__ == "__main__":
    main()