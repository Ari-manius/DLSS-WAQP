"""
Hyperparameter optimization for GNN models using Optuna.
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import argparse
import os
import numpy as np
import json
from datetime import datetime

# Import utilities
from utils.create_split_masks import create_split_masks_regression
from utils.GNN_model import ImprovedGNN, ResidualGCN, GraphAttentionNet, ResidualGraphSAGE
from utils.earlyStopping import EarlyStopping
from utils.initialize_weights import initialize_weights
from utils.loss_functions import FocalLoss, ClassBalancedFocalLoss, WeightedCrossEntropyLoss
from utils.feature_engineering import add_node_degree_features, compute_graph_statistics
from utils.cross_validation import cross_validate_model, print_cv_results


def objective(trial, data, model_type='improved_gnn', task_type='classification', device='cpu', 
              n_splits=3, epochs=100):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        data: PyTorch Geometric Data object
        model_type: Type of model to optimize
        task_type: 'classification' or 'regression'
        device: Device to use for training
        n_splits: Number of CV folds
        epochs: Number of epochs for each trial
    
    Returns:
        Mean validation score across folds
    """
    
    # Hyperparameter search space
    params = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-1),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    }
    
    # Model-specific parameters
    if model_type == 'gat':
        params['heads'] = trial.suggest_categorical('heads', [2, 4, 8])
    
    # Loss function parameters (for classification)
    if task_type == 'classification':
        loss_type = trial.suggest_categorical('loss_type', ['focal', 'weighted_ce', 'ce'])
        if loss_type == 'focal':
            params['focal_alpha'] = trial.suggest_uniform('focal_alpha', 0.5, 2.0)
            params['focal_gamma'] = trial.suggest_uniform('focal_gamma', 1.0, 3.0)
    
    # Model parameters
    input_dim = data.num_node_features
    output_dim = len(data.y.unique()) if task_type == 'classification' else 1
    
    model_params = {
        'hidden_dim': params['hidden_dim'],
        'num_layers': params['num_layers'],
        'dropout': params['dropout']
    }
    
    if model_type == 'gat':
        model_params['heads'] = params['heads']
    
    # Training parameters
    train_params = {
        'optimizer': {
            'lr': params['lr'],
            'weight_decay': params['weight_decay']
        },
        'epochs': epochs
    }
    
    # Model selection
    if model_type == 'improved_gnn':
        model_class = ImprovedGNN
    elif model_type == 'residual_gcn':
        model_class = ResidualGCN
    elif model_type == 'gat':
        model_class = GraphAttentionNet
    elif model_type == 'residual_sage':
        model_class = ResidualGraphSAGE
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        # Perform cross-validation
        cv_results = cross_validate_model(
            model_class=model_class,
            data=data,
            model_params=model_params,
            train_params=train_params,
            n_splits=n_splits,
            device=device,
            verbose=False
        )
        
        # Return objective value
        if task_type == 'classification':
            return cv_results['mean_val_accuracy']
        else:
            return -cv_results['mean_val_loss']  # Minimize loss (maximize negative loss)
            
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('-inf') if task_type == 'classification' else float('inf')


def optimize_hyperparameters(data, model_type='improved_gnn', n_trials=100, 
                           n_splits=3, epochs_per_trial=50, device='cpu',
                           study_name=None, storage=None):
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        data: PyTorch Geometric Data object
        model_type: Type of model to optimize
        n_trials: Number of optimization trials
        n_splits: Number of CV folds per trial
        epochs_per_trial: Epochs to train each model
        device: Device to use
        study_name: Name for the study
        storage: Storage backend for study
    
    Returns:
        Best parameters and study object
    """
    
    # Determine task type
    task_type = 'classification' if len(data.y.unique()) > 1 else 'regression'
    direction = 'maximize' if task_type == 'classification' else 'minimize'
    
    print(f"Starting hyperparameter optimization for {model_type}")
    print(f"Task type: {task_type}")
    print(f"Number of trials: {n_trials}")
    print(f"CV folds: {n_splits}")
    print(f"Epochs per trial: {epochs_per_trial}")
    print(f"Device: {device}")
    
    # Create study
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{model_type}_optimization_{timestamp}"
    
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    
    # Optimization callback
    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Trial {trial.number}: {trial.value:.4f}")
            if len(study.trials) % 10 == 0:
                print(f"Best value so far: {study.best_value:.4f}")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, data, model_type, task_type, device, n_splits, epochs_per_trial
        ),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params, study


def train_final_model(data, best_params, model_type, device, epochs=200):
    """
    Train final model with best hyperparameters.
    """
    print(f"\nTraining final model with best parameters...")
    
    # Setup data splits
    train_mask, val_mask, test_mask = create_split_masks_regression(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Model setup
    input_dim = data.num_node_features
    task_type = 'classification' if len(data.y.unique()) > 1 else 'regression'
    output_dim = len(data.y.unique()) if task_type == 'classification' else 1
    
    # Extract model parameters
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': best_params['hidden_dim'],
        'output_dim': output_dim,
        'num_layers': best_params['num_layers'],
        'dropout': best_params['dropout']
    }
    
    if model_type == 'gat' and 'heads' in best_params:
        model_params['heads'] = best_params['heads']
    
    # Initialize model
    if model_type == 'improved_gnn':
        model = ImprovedGNN(**model_params)
    elif model_type == 'residual_gcn':
        model = ResidualGCN(**model_params)
    elif model_type == 'gat':
        model = GraphAttentionNet(**model_params)
    elif model_type == 'residual_sage':
        model = ResidualGraphSAGE(**model_params)
    
    model = model.to(device)
    initialize_weights(model)
    
    # Setup loss and optimizer
    if task_type == 'classification':
        if best_params.get('loss_type') == 'focal':
            criterion = FocalLoss(
                alpha=best_params.get('focal_alpha', 1.0),
                gamma=best_params.get('focal_gamma', 2.0)
            )
        elif best_params.get('loss_type') == 'weighted_ce':
            criterion = WeightedCrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    
    scheduler = OneCycleLR(
        optimizer, max_lr=best_params['lr'], 
        epochs=epochs, steps_per_epoch=1, 
        pct_start=0.1, anneal_strategy='cos'
    )
    
    # Training
    data = data.to(device)
    early_stopper = EarlyStopping(patience=50, min_delta=1e-4)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        output = model(data)
        train_loss = criterion(output[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            output = model(data)
            val_loss = criterion(output[data.val_mask], data.y[data.val_mask])
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping
        early_stopper(val_loss.item(), model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if epoch % 20 == 0:
            if task_type == 'classification':
                _, val_pred = torch.max(output[data.val_mask], 1)
                val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
                print(f"Epoch {epoch:3d}: Train Loss {train_loss.item():.4f}, "
                      f"Val Loss {val_loss.item():.4f}, Val Acc {val_acc:.4f}")
            else:
                print(f"Epoch {epoch:3d}: Train Loss {train_loss.item():.4f}, "
                      f"Val Loss {val_loss.item():.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        output = model(data)
        
        if task_type == 'classification':
            _, test_pred = torch.max(output[data.test_mask], 1)
            test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()
            print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        else:
            test_loss = criterion(output[data.test_mask], data.y[data.test_mask]).item()
            print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for GNN models")
    parser.add_argument('--data_file', type=str, required=True,
                       help='Base name of the .pt file (without extension)')
    parser.add_argument('--model_type', type=str, default='improved_gnn',
                       choices=['improved_gnn', 'residual_gcn', 'gat', 'residual_sage'],
                       help='Model architecture to optimize')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--n_splits', type=int, default=3,
                       help='Number of CV folds')
    parser.add_argument('--epochs_per_trial', type=int, default=50,
                       help='Epochs per trial during optimization')
    parser.add_argument('--final_epochs', type=int, default=200,
                       help='Epochs for final model training')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        # elif torch.backends.mps.is_available():
        #     device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data
    data_path = os.path.join("data", args.data_file + ".pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = torch.load(data_path, weights_only=False)
    
    compute_graph_statistics(data)
    
    # Run optimization
    best_params, study = optimize_hyperparameters(
        data=data,
        model_type=args.model_type,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        epochs_per_trial=args.epochs_per_trial,
        device=device
    )
    
    # Save results
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{args.model_type}_{args.data_file}_{timestamp}.json")
    
    results = {
        'best_params': best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'model_type': args.model_type,
        'data_file': args.data_file,
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Train final model
    final_model = train_final_model(
        data=data,
        best_params=best_params,
        model_type=args.model_type,
        device=device,
        epochs=args.final_epochs
    )
    
    # Save final model
    model_file = os.path.join(results_dir, f"best_{args.model_type}_{args.data_file}_{timestamp}.pt")
    torch.save(final_model.state_dict(), model_file)
    print(f"Final model saved to: {model_file}")


if __name__ == "__main__":
    main()