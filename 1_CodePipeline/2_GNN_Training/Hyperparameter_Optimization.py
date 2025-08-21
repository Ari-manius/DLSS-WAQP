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
import logging
import sys
from tqdm import tqdm
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

# Force clean MPS environment variable immediately at import time
if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
    del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
    print("🔧 Cleaned problematic MPS environment variable at startup")

# Import utilities
from utils.create_split_masks import create_split_masks_regression
from utils.GNN_model import ImprovedGNN, ResidualGCN, GraphAttentionNet, ResidualGraphSAGE, MLPBaseline
from utils.earlyStopping import EarlyStopping
from utils.initialize_weights import initialize_weights
from utils.loss_functions import FocalLoss, ClassBalancedFocalLoss, WeightedCrossEntropyLoss, smart_oversample_indices
from utils.feature_engineering import compute_graph_statistics
from utils.cross_validation import cross_validate_model, print_cv_results
from sklearn.metrics import f1_score, precision_recall_fscore_support

def setup_logging(model_type, verbose=True):
    """Setup logging for optimization progress tracking."""
    log_dir = "optimization_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_type}_optimization_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if verbose else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting optimization for {model_type}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def save_trial_progress(trial_num, trial_value, params, model_type, best_so_far=None):
    """Save individual trial progress to a JSON file."""
    progress_dir = "optimization_progress"
    os.makedirs(progress_dir, exist_ok=True)
    
    progress_file = os.path.join(progress_dir, f"{model_type}_progress.json")
    
    # Load existing progress or create new
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
        except:
            progress_data = {"trials": [], "metadata": {}}
    else:
        progress_data = {"trials": [], "metadata": {}}
    
    # Add current trial
    trial_data = {
        "trial_num": trial_num,
        "value": trial_value,
        "params": params,
        "timestamp": datetime.now().isoformat(),
        "is_best": best_so_far is None or trial_value > best_so_far
    }
    
    progress_data["trials"].append(trial_data)
    progress_data["metadata"].update({
        "model_type": model_type,
        "total_trials": len(progress_data["trials"]),
        "best_value": max([t["value"] for t in progress_data["trials"]] + [0]),
        "last_updated": datetime.now().isoformat()
    })
    
    # Save progress
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    return progress_file


def objective(trial, data, model_type='improved_gnn', task_type='classification', device='cpu', 
              n_splits=3, epochs=100, logger=None):
    """
    Enhanced Optuna objective function with detailed logging.
    
    Args:
        trial: Optuna trial object
        data: PyTorch Geometric Data object
        model_type: Type of model to optimize
        task_type: 'classification' or 'regression'
        device: Device to use for training
        n_splits: Number of CV folds
        epochs: Number of epochs for each trial
        logger: Logger instance for progress tracking
    
    Returns:
        Mean validation score across folds
    """
    if logger:
        logger.info(f"🔄 Starting trial {trial.number}")
        logger.info(f"   Device: {device}, Task: {task_type}, CV folds: {n_splits}")
    
    # Hyperparameter search space with model-specific constraints
    params = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.7),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    }
    
    # Model-specific parameters and constraints
    if model_type == 'gat':
        params['heads'] = trial.suggest_categorical('heads', [2, 4, 8])
        # GAT requires lower learning rates due to attention mechanism sensitivity
        params['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    else:
        # Standard learning rate range for other models
        params['lr'] = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    
    # Enhanced loss function parameters for class imbalance
    if task_type == 'classification':
        loss_type = trial.suggest_categorical('loss_type', ['class_balanced_focal', 'focal', 'weighted_ce'])
        
        if loss_type == 'focal':
            params['focal_alpha'] = trial.suggest_float('focal_alpha', 1.0, 3.0)
            params['focal_gamma'] = trial.suggest_float('focal_gamma', 2.0, 4.0)
        elif loss_type == 'class_balanced_focal':
            params['cb_focal_beta'] = trial.suggest_float('cb_focal_beta', 0.999, 0.9999)
            params['cb_focal_gamma'] = trial.suggest_float('cb_focal_gamma', 2.0, 4.0)
            params['min_class_boost'] = trial.suggest_float('min_class_boost', 2.0, 5.0)
        
        # Oversampling strategy parameters
        params['oversample_strategy'] = trial.suggest_categorical('oversample_strategy', ['balanced', 'boosted'])
        params['min_samples_factor'] = trial.suggest_int('min_samples_factor', 2, 4)
    
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
    
    # Add class imbalance handling parameters for classification
    if task_type == 'classification':
        train_params['class_balance'] = {
            'oversample_strategy': params.get('oversample_strategy', 'boosted'),
            'min_samples_factor': params.get('min_samples_factor', 3),
            'max_total_samples': 20000  # Memory-safe limit
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
    elif model_type == 'mlp':
        model_class = MLPBaseline
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if logger:
        logger.info(f"   📊 Parameters: {params}")
    
    try:
        # Perform cross-validation with progress tracking
        if logger:
            logger.info(f"   🔬 Running {n_splits}-fold cross-validation...")
            
        cv_results = cross_validate_model(
            model_class=model_class,
            data=data,
            model_params=model_params,
            train_params=train_params,
            n_splits=n_splits,
            device=device,
            verbose=False
        )
        
        # Calculate objective value focused on minority class performance
        if task_type == 'classification':
            # Use weighted average of overall accuracy and minority class F1
            overall_acc = cv_results['mean_val_accuracy']
            
            # If minority class metrics are available, combine them
            if 'minority_f1' in cv_results:
                minority_f1 = cv_results['minority_f1']
                # Weight: 60% minority F1, 40% overall accuracy
                objective_value = 0.6 * minority_f1 + 0.4 * overall_acc
                
                if logger:
                    logger.info(f"   📈 Results: Overall Acc={overall_acc:.4f}, Minority F1={minority_f1:.4f}, Combined={objective_value:.4f}")
            else:
                objective_value = overall_acc
                if logger:
                    logger.info(f"   📈 Results: Overall Acc={objective_value:.4f}")
        else:
            objective_value = -cv_results['mean_val_loss']
            if logger:
                logger.info(f"   📈 Results: Validation Loss={-objective_value:.4f}")
        
        # Save trial progress
        save_trial_progress(trial.number, objective_value, params, model_type)
        
        if logger:
            logger.info(f"   ✅ Trial {trial.number} completed successfully")
            
        return objective_value
            
    except Exception as e:
        error_msg = f"Trial {trial.number} failed: {str(e)}"
        if logger:
            logger.error(f"   ❌ {error_msg}")
        else:
            print(error_msg)
        return float('-inf') if task_type == 'classification' else float('inf')


def optimize_hyperparameters(data, model_type='improved_gnn', n_trials=100, 
                           n_splits=3, epochs_per_trial=50, device='cpu',
                           study_name=None, storage=None, verbose=True):
    """
    Optimize hyperparameters using Optuna with comprehensive progress tracking.
    
    Args:
        data: PyTorch Geometric Data object
        model_type: Type of model to optimize
        n_trials: Number of optimization trials
        n_splits: Number of CV folds per trial
        epochs_per_trial: Epochs to train each model
        device: Device to use
        study_name: Name for the study
        storage: Storage backend for study
        verbose: Enable detailed logging
    
    Returns:
        Best parameters and study object
    """
    
    # Setup comprehensive logging
    logger = setup_logging(model_type, verbose)
    
    # Determine task type
    task_type = 'classification' if len(data.y.unique()) > 1 else 'regression'
    direction = 'maximize' if task_type == 'classification' else 'minimize'
    
    logger.info(f"🚀 Starting hyperparameter optimization for {model_type}")
    logger.info(f"📋 Configuration:")
    logger.info(f"   Task type: {task_type}")
    logger.info(f"   Trials: {n_trials}")
    logger.info(f"   CV folds: {n_splits}")
    logger.info(f"   Epochs per trial: {epochs_per_trial}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Direction: {direction}")
    
    # Print class distribution for classification tasks
    if task_type == 'classification':
        unique, counts = torch.unique(data.y, return_counts=True)
        logger.info(f"📊 Class distribution:")
        for cls, count in zip(unique, counts):
            percentage = 100 * count / len(data.y)
            logger.info(f"   Class {cls}: {count} samples ({percentage:.1f}%)")
    
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
    
    # Enhanced optimization callback with detailed progress tracking
    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            progress_pct = (trial.number + 1) / n_trials * 100
            logger.info(f"🎯 Trial {trial.number + 1}/{n_trials} ({progress_pct:.1f}%): Score = {trial.value:.4f}")
            
            if len(study.trials) % 5 == 0:  # Report every 5 trials
                best_value = study.best_value
                best_trial_num = study.best_trial.number
                logger.info(f"📊 Progress Update - {len(study.trials)} trials completed")
                logger.info(f"   🏆 Best score so far: {best_value:.4f} (Trial #{best_trial_num})")
                
                # Show ETA
                if len(study.trials) >= 5:
                    import time
                    trials_completed = len(study.trials)
                    total_time = time.time() - study.trials[0].datetime_start.timestamp()
                    avg_time_per_trial = total_time / trials_completed
                    remaining_trials = n_trials - trials_completed
                    eta_seconds = remaining_trials * avg_time_per_trial
                    eta_minutes = eta_seconds / 60
                    logger.info(f"   ⏱️  ETA: {eta_minutes:.1f} minutes remaining")
        
        elif trial.state == optuna.trial.TrialState.FAIL:
            logger.warning(f"⚠️  Trial {trial.number} failed")
    
    logger.info(f"🔍 Starting optimization with {n_trials} trials...")
    
    # Run optimization with progress tracking
    study.optimize(
        lambda trial: objective(
            trial, data, model_type, task_type, device, n_splits, epochs_per_trial, logger
        ),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=False  # Using custom logging instead
    )
    
    # Final results with detailed logging
    logger.info(f"\n🎉 OPTIMIZATION COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info(f"📋 Summary:")
    logger.info(f"   Total trials: {len(study.trials)}")
    logger.info(f"   Best trial: #{study.best_trial.number}")
    logger.info(f"   Best value: {study.best_value:.4f}")
    logger.info(f"🏆 Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"   {key}: {value}")
    
    # Also print to console for runner script
    print(f"\n✅ {model_type} optimization completed!")
    print(f"   Best score: {study.best_value:.4f}")
    print(f"   Total trials: {len(study.trials)}")
    
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
    elif model_type == 'mlp':
        model = MLPBaseline(**model_params)
    
    # Handle MPS device issues
    if device.type == 'mps':
        import os
        torch.mps.empty_cache()
        # Remove any problematic environment variables
        if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
            del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
    
    try:
        model = model.to(device)
    except RuntimeError as e:
        if "watermark ratio" in str(e):
            print(f"⚠ MPS watermark error: {e}")
            print("⚠ Falling back to CPU for stability")
            device = torch.device('cpu')
            model = model.to(device)
        else:
            raise e
            
    initialize_weights(model)
    
    # Setup enhanced loss function with class imbalance handling
    if task_type == 'classification':
        loss_type = best_params.get('loss_type', 'class_balanced_focal')
        
        if loss_type == 'focal':
            criterion = FocalLoss(
                alpha=best_params.get('focal_alpha', 2.0),
                gamma=best_params.get('focal_gamma', 3.0)
            )
        elif loss_type == 'class_balanced_focal':
            criterion = ClassBalancedFocalLoss(
                beta=best_params.get('cb_focal_beta', 0.9999),
                gamma=best_params.get('cb_focal_gamma', 3.0),
                min_class_boost=best_params.get('min_class_boost', 3.0)
            )
        elif loss_type == 'weighted_ce':
            criterion = WeightedCrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
            
        # Print class distribution for monitoring
        train_labels = data.y[data.train_mask]
        unique, counts = torch.unique(train_labels, return_counts=True)
        print("Class distribution in training set:")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples ({100*count/len(train_labels):.1f}%)")
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
        # Training with enhanced oversampling for classification
        model.train()
        optimizer.zero_grad()
        
        # Use GraphSAINT sampling for memory efficiency
        if hasattr(data, 'train_mask') and data.train_mask.sum() > 10000:
            # Sample subgraph for large datasets
            train_indices = torch.where(data.train_mask)[0].to(data.x.device)
            if len(train_indices) == 0:
                # Fallback if no training data
                output = model(data)
                train_mask_sample = data.train_mask
            else:
                sample_size = min(8000, len(train_indices))  # Limit sample size
                sample_size = max(1, sample_size)  # Ensure at least 1 sample
                sampled_indices = train_indices[torch.randperm(len(train_indices), device=train_indices.device)[:sample_size]]
                
                # Create subgraph
                edge_index, _ = subgraph(sampled_indices, data.edge_index, relabel_nodes=True)
                x_sample = data.x[sampled_indices]
                y_sample = data.y[sampled_indices]
                
                # Create temporary data object
                sample_data = Data(x=x_sample, edge_index=edge_index, y=y_sample)
                output = model(sample_data)
                train_mask_sample = torch.ones(len(sampled_indices), dtype=torch.bool, device=output.device)
        else:
            output = model(data)
            train_mask_sample = data.train_mask
        
        # Apply loss calculation based on sampling strategy
        if task_type == 'classification':
            if hasattr(data, 'train_mask') and data.train_mask.sum() > 10000:
                # For sampled data, use the sample
                train_loss = criterion(output[train_mask_sample], y_sample)
            else:
                # For smaller datasets, can still use oversampling
                train_indices = torch.where(data.train_mask)[0].to(data.x.device)
                train_labels = data.y[train_indices]
                
                if len(torch.unique(train_labels)) > 1:
                    oversample_strategy = best_params.get('oversample_strategy', 'boosted')
                    min_samples_factor = best_params.get('min_samples_factor', 3)
                    
                    oversampled_train_indices = smart_oversample_indices(
                        train_indices, train_labels, 
                        strategy=oversample_strategy, 
                        min_samples_factor=min_samples_factor,
                        max_total_samples=20000  # Memory-safe limit
                    )
                    train_loss = criterion(output[oversampled_train_indices], data.y[oversampled_train_indices])
                else:
                    train_loss = criterion(output[data.train_mask], data.y[data.train_mask])
        else:
            # Regression: use appropriate mask based on sampling
            if hasattr(data, 'train_mask') and data.train_mask.sum() > 10000:
                train_loss = criterion(output[train_mask_sample], y_sample)
            else:
                train_loss = criterion(output[data.train_mask], data.y[data.train_mask])
            
        train_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Use sampling for validation too if dataset is large
            if hasattr(data, 'val_mask') and data.val_mask.sum() > 5000:
                val_indices = torch.where(data.val_mask)[0].to(data.x.device)
                sample_size = min(4000, len(val_indices))
                sampled_val_indices = val_indices[torch.randperm(len(val_indices), device=val_indices.device)[:sample_size]]
                
                edge_index, _ = subgraph(sampled_val_indices, data.edge_index, relabel_nodes=True)
                x_val_sample = data.x[sampled_val_indices]
                y_val_sample = data.y[sampled_val_indices]
                
                val_sample_data = Data(x=x_val_sample, edge_index=edge_index, y=y_val_sample)
                output = model(val_sample_data)
                val_mask_sample = torch.ones(len(sampled_val_indices), dtype=torch.bool, device=output.device)
                val_loss = criterion(output[val_mask_sample], y_val_sample)
            else:
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
                # Use correct validation indices based on whether we sampled or not
                if hasattr(data, 'val_mask') and data.val_mask.sum() > 5000:
                    # We used sampling, so compare with sampled data
                    _, val_pred = torch.max(output[val_mask_sample], 1)
                    val_acc = (val_pred == y_val_sample).float().mean().item()
                else:
                    # No sampling, use original data
                    _, val_pred = torch.max(output[data.val_mask], 1)
                    val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
                print(f"Epoch {epoch:3d}: Train Loss {train_loss.item():.4f}, "
                      f"Val Loss {val_loss.item():.4f}, Val Acc {val_acc:.4f}")
            else:
                print(f"Epoch {epoch:3d}: Train Loss {train_loss.item():.4f}, "
                      f"Val Loss {val_loss.item():.4f}")
    
    # Enhanced final evaluation with minority class metrics
    model.eval()
    with torch.no_grad():
        # Use sampling for final evaluation too if dataset is large
        if hasattr(data, 'test_mask') and data.test_mask.sum() > 5000:
            test_indices = torch.where(data.test_mask)[0].to(data.x.device)
            sample_size = min(4000, len(test_indices))
            sampled_test_indices = test_indices[torch.randperm(len(test_indices), device=test_indices.device)[:sample_size]]
            
            edge_index, _ = subgraph(sampled_test_indices, data.edge_index, relabel_nodes=True)
            x_test_sample = data.x[sampled_test_indices]
            y_test_sample = data.y[sampled_test_indices]
            
            test_sample_data = Data(x=x_test_sample, edge_index=edge_index, y=y_test_sample)
            output = model(test_sample_data)
            test_mask_sample = torch.ones(len(sampled_test_indices), dtype=torch.bool, device=output.device)
            
            if task_type == 'classification':
                _, test_pred = torch.max(output[test_mask_sample], 1)
                test_targets = y_test_sample
        else:
            output = model(data)
            if task_type == 'classification':
                _, test_pred = torch.max(output[data.test_mask], 1)
                test_targets = data.y[data.test_mask]
        
        if task_type == 'classification':
            
            test_acc = (test_pred == test_targets).float().mean().item()
            
            # Calculate per-class metrics
            unique_classes = torch.unique(test_targets)
            class_accuracies = []
            
            print(f"\nFinal Test Results:")
            print(f"Overall Accuracy: {test_acc:.4f}")
            print("\nPer-class Performance:")
            
            for class_idx in unique_classes:
                class_mask = test_targets == class_idx
                class_pred = test_pred[class_mask]
                class_true = test_targets[class_mask]
                class_acc = (class_pred == class_true).float().mean().item()
                class_accuracies.append(class_acc)
                class_count = class_mask.sum().item()
                print(f"  Class {class_idx}: {class_acc:.4f} (n={class_count})")
            
            # Identify and highlight smallest class performance
            test_labels_np = test_targets.cpu().numpy()
            test_pred_np = test_pred.cpu().numpy()
            
            # Calculate F1 scores per class
            from sklearn.metrics import f1_score, classification_report
            
            try:
                f1_scores = f1_score(test_labels_np, test_pred_np, average=None, zero_division=0)
                macro_f1 = f1_score(test_labels_np, test_pred_np, average='macro', zero_division=0)
                weighted_f1 = f1_score(test_labels_np, test_pred_np, average='weighted', zero_division=0)
                
                print(f"\nF1 Scores:")
                print(f"Macro F1: {macro_f1:.4f}")
                print(f"Weighted F1: {weighted_f1:.4f}")
                print("Per-class F1:")
                for i, (class_idx, f1) in enumerate(zip(unique_classes, f1_scores)):
                    print(f"  Class {class_idx}: {f1:.4f}")
                
                # Find smallest class
                unique, counts = torch.unique(test_targets, return_counts=True)
                smallest_class_idx = unique[torch.argmin(counts)].item()
                smallest_class_f1 = f1_scores[list(unique_classes.cpu().numpy()).index(smallest_class_idx)]
                
                print(f"\nSmallest Class Performance:")
                print(f"Class {smallest_class_idx} F1 Score: {smallest_class_f1:.4f}")
                
            except Exception as e:
                print(f"Error calculating detailed metrics: {e}")
                
        else:
            test_loss = criterion(output[data.test_mask], data.y[data.test_mask]).item()
            print(f"\nFinal Test Loss: {test_loss:.4f}")
    
    # Memory cleanup for MPS
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for GNN models")
    parser.add_argument('--data_file', type=str, required=True,
                       help='Base name of the .pt file (without extension)')
    parser.add_argument('--model_type', type=str, default='improved_gnn',
                       choices=['improved_gnn', 'residual_gcn', 'gat', 'residual_sage', 'mlp'],
                       help='Model architecture to optimize')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--n_splits', type=int, default=3,
                       help='Number of CV folds')
    parser.add_argument('--epochs_per_trial', type=int, default=50,
                       help='Epochs per trial during optimization')
    parser.add_argument('--final_epochs', type=int, default=200,
                       help='Epochs for final model training')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
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