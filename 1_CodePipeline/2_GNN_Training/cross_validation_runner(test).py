#!/usr/bin/env python3
"""
Cross-Validation GNN Training Runner
Implements k-fold cross validation with multiple models and ensemble predictions.
"""

import subprocess
import sys
import os
import json
import glob
import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from collections import defaultdict
import pickle

def get_latest_optimization_results(model_type, data_file):
    """Get the latest optimization results for a specific model."""
    results_dir = "optimization_results"
    
    if not os.path.exists(results_dir):
        return None
        
    pattern = os.path.join(results_dir, f"{model_type}_*{data_file}*.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        return None
        
    latest_file = max(result_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        return results.get('best_params', {})
    except Exception as e:
        print(f"⚠ Error loading optimization results for {model_type}: {e}")
        return None

def get_fallback_params(model_type):
    """Get fallback parameters if optimization results aren't available."""
    fallback_params = {
        'improved_gnn': {
            'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.4,
            'lr': 0.01, 'weight_decay': 1e-4, 'loss_type': 'class_balanced_focal',
            'cb_focal_beta': 0.9999, 'cb_focal_gamma': 3.0, 'min_class_boost': 3.0,
            'oversample_strategy': 'boosted', 'min_samples_factor': 2
        },
        'residual_gcn': {
            'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.3,
            'lr': 0.01, 'weight_decay': 1e-4, 'loss_type': 'class_balanced_focal',
            'cb_focal_beta': 0.9999, 'cb_focal_gamma': 3.0, 'min_class_boost': 3.0,
            'oversample_strategy': 'boosted', 'min_samples_factor': 2
        },
        'gat': {
            'hidden_dim': 64, 'num_layers': 2, 'heads': 4, 'dropout': 0.5,
            'lr': 0.005, 'weight_decay': 1e-4, 'loss_type': 'focal'
        },
        'residual_sage': {
            'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.4,
            'lr': 0.01, 'weight_decay': 1e-4, 'loss_type': 'class_balanced_focal',
            'cb_focal_beta': 0.9999, 'cb_focal_gamma': 3.0, 'min_class_boost': 3.0,
            'oversample_strategy': 'boosted', 'min_samples_factor': 2
        },
        'mlp': {
            'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.5,
            'lr': 0.01, 'weight_decay': 1e-3, 'loss_type': 'weighted_ce'
        },
        'mlp_non_network': {
            'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.5,
            'lr': 0.01, 'weight_decay': 1e-3, 'loss_type': 'weighted_ce'
        }
    }
    return fallback_params.get(model_type, {})

def create_cv_splits(data_file, k_folds=5, stratify=True):
    """Create cross-validation splits and save them."""
    from utils.load_graph_data import load_graph_data
    
    # Load the data to get labels for stratification
    print(f"Loading data from {data_file} for CV split creation...")
    data = load_graph_data(data_file)
    
    if data is None:
        raise ValueError(f"Could not load data from {data_file}")
    
    # Get labels
    y = data.y.cpu().numpy()
    n_samples = len(y)
    
    # Create splits
    if stratify and len(np.unique(y)) > 1:
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        print(f"Using stratified {k_folds}-fold cross-validation")
    else:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        print(f"Using {k_folds}-fold cross-validation")
    
    splits = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n_samples), y)):
        splits.append({
            'fold': fold,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist()
        })
    
    # Save splits
    splits_dir = "cv_splits"
    os.makedirs(splits_dir, exist_ok=True)
    splits_file = os.path.join(splits_dir, f"{data_file}_cv_splits_k{k_folds}.json")
    
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✓ Saved {k_folds} CV splits to {splits_file}")
    
    # Print split statistics
    for fold_data in splits:
        fold = fold_data['fold']
        train_size = len(fold_data['train_indices'])
        val_size = len(fold_data['val_indices'])
        print(f"  Fold {fold}: Train={train_size}, Val={val_size}")
    
    return splits_file

def build_cv_training_command(model_type, data_file, fold, splits_file, epochs=100):
    """Build training command for cross-validation."""
    opt_params = get_latest_optimization_results(model_type, data_file)
    
    if opt_params:
        print(f"✓ Using optimized parameters for {model_type}")
        params = opt_params
    else:
        print(f"⚠ Using fallback parameters for {model_type}")
        params = get_fallback_params(model_type)
    
    actual_model_type = 'mlp' if model_type == 'mlp_non_network' else model_type
    cmd = f"python3 cv_training.py --data_file {data_file} --model_type {actual_model_type}"
    cmd += f" --cv_splits_file {splits_file} --fold {fold}"
    
    # Add optimized parameters
    if 'hidden_dim' in params:
        cmd += f" --hidden_dim {params['hidden_dim']}"
    if 'num_layers' in params:
        cmd += f" --num_layers {params['num_layers']}"
    if 'dropout' in params:
        cmd += f" --dropout {params['dropout']}"
    if 'lr' in params:
        cmd += f" --lr {params['lr']}"
    if 'weight_decay' in params:
        cmd += f" --weight_decay {params['weight_decay']}"
    if 'loss_type' in params:
        cmd += f" --loss_type {params['loss_type']}"
    
    # Add loss function parameters
    if 'cb_focal_beta' in params:
        cmd += f" --cb_focal_beta {params['cb_focal_beta']}"
    if 'cb_focal_gamma' in params:
        cmd += f" --cb_focal_gamma {params['cb_focal_gamma']}"
    if 'min_class_boost' in params:
        cmd += f" --min_class_boost {params['min_class_boost']}"
    if 'oversample_strategy' in params:
        cmd += f" --oversample_strategy {params['oversample_strategy']}"
    if 'min_samples_factor' in params:
        cmd += f" --min_samples_factor {params['min_samples_factor']}"
    
    # GAT-specific parameters
    if model_type == 'gat' and 'heads' in params:
        cmd += f" --heads {params['heads']}"
    
    # Memory optimization
    if model_type == 'gat':
        cmd += " --use_graphsaint --batch_size 2048 --walk_length 2 --num_steps 8"
    else:
        cmd += " --use_graphsaint --batch_size 8192 --walk_length 2 --num_steps 8"
    
    cmd += f" --epochs {epochs} --device auto"
    
    return cmd

def run_command(command):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {command}")
    print(f"{'='*80}")
    
    try:
        cmd_clean = ' '.join(command.split())
        cmd_parts = cmd_clean.split()
        result = subprocess.run(cmd_parts, check=True, capture_output=True, text=True)
        print(f"✓ Command completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr

def aggregate_cv_results(model_types, data_file, k_folds):
    """Aggregate cross-validation results across all folds."""
    results_dir = "cv_results"
    aggregated_results = {}
    
    for model_type in model_types:
        model_results = {
            'fold_scores': [],
            'mean_score': 0,
            'std_score': 0,
            'predictions': [],
            'true_labels': []
        }
        
        # Collect results from all folds
        for fold in range(k_folds):
            result_file = os.path.join(results_dir, f"{model_type}_{data_file}_fold_{fold}_results.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    fold_result = json.load(f)
                    model_results['fold_scores'].append(fold_result.get('test_accuracy', 0))
        
        if model_results['fold_scores']:
            model_results['mean_score'] = np.mean(model_results['fold_scores'])
            model_results['std_score'] = np.std(model_results['fold_scores'])
        
        aggregated_results[model_type] = model_results
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_file = os.path.join(results_dir, f"cv_aggregated_results_{timestamp}.json")
    
    with open(agg_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    return aggregated_results, agg_file

def create_ensemble_predictions(model_types, data_file, k_folds):
    """Create ensemble predictions by averaging model outputs."""
    results_dir = "cv_results"
    ensemble_predictions = defaultdict(list)
    ensemble_true_labels = []
    
    # Load predictions from all models and folds
    for model_type in model_types:
        for fold in range(k_folds):
            pred_file = os.path.join(results_dir, f"{model_type}_{data_file}_fold_{fold}_predictions.pkl")
            
            if os.path.exists(pred_file):
                with open(pred_file, 'rb') as f:
                    pred_data = pickle.load(f)
                    ensemble_predictions[model_type].append(pred_data['predictions'])
                    if not ensemble_true_labels and 'true_labels' in pred_data:
                        ensemble_true_labels = pred_data['true_labels']
    
    # Average predictions across folds for each model
    model_avg_predictions = {}
    for model_type, fold_predictions in ensemble_predictions.items():
        if fold_predictions:
            model_avg_predictions[model_type] = np.mean(fold_predictions, axis=0)
    
    # Create final ensemble by averaging across models
    if model_avg_predictions:
        final_ensemble = np.mean(list(model_avg_predictions.values()), axis=0)
        
        # Save ensemble results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_file = os.path.join(results_dir, f"ensemble_predictions_{timestamp}.pkl")
        
        ensemble_data = {
            'ensemble_predictions': final_ensemble,
            'model_predictions': model_avg_predictions,
            'true_labels': ensemble_true_labels
        }
        
        with open(ensemble_file, 'rb') as f:
            pickle.dump(ensemble_data, f)
        
        print(f"✓ Ensemble predictions saved to {ensemble_file}")
        return ensemble_file
    
    return None

def main():
    """Execute cross-validation training pipeline."""
    
    print("Cross-Validation GNN Training Pipeline")
    print("=" * 80)
    
    # Configuration
    data_file = "data_quantile_Target_QC_aggcat"
    non_network_data_file = "data_nonnetwork_quantile_Target_QC_aggcat"
    model_types = ['mlp_non_network', 'improved_gnn', 'residual_gcn', 'residual_sage', 'gat', 'mlp']
    epochs = 100  # Reduced for CV
    k_folds = 5
    
    # Create results directory
    os.makedirs("cv_results", exist_ok=True)
    
    print(f"📋 Configuration:")
    print(f"  Models: {', '.join(model_types)}")
    print(f"  K-folds: {k_folds}")
    print(f"  Epochs per fold: {epochs}")
    print("=" * 80)
    
    # Step 1: Create CV splits
    print(f"\nStep 1: Creating {k_folds}-fold cross-validation splits")
    splits_file = create_cv_splits(data_file, k_folds, stratify=True)
    
    # Step 2: Train models with cross-validation
    print(f"\nStep 2: Training models with cross-validation")
    total_runs = len(model_types) * k_folds
    completed_runs = 0
    failed_runs = []
    
    for model_type in model_types:
        print(f"\n🔄 Training {model_type} with {k_folds}-fold CV")
        print("-" * 60)
        
        current_data_file = non_network_data_file if model_type == 'mlp_non_network' else data_file
        
        for fold in range(k_folds):
            print(f"\n  Fold {fold + 1}/{k_folds}")
            
            # Build command
            command = build_cv_training_command(model_type, current_data_file, fold, splits_file, epochs)
            
            # Execute training
            success, output = run_command(command)
            
            if success:
                completed_runs += 1
                print(f"  ✓ {model_type} fold {fold} completed")
            else:
                failed_runs.append(f"{model_type}_fold_{fold}")
                print(f"  ✗ {model_type} fold {fold} failed")
    
    # Step 3: Aggregate results
    print(f"\nStep 3: Aggregating cross-validation results")
    aggregated_results, agg_file = aggregate_cv_results(model_types, data_file, k_folds)
    
    # Step 4: Create ensemble predictions
    print(f"\nStep 4: Creating ensemble predictions")
    ensemble_file = create_ensemble_predictions(model_types, data_file, k_folds)
    
    # Final Summary
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed_runs}/{total_runs}")
    print(f"Success rate: {completed_runs/total_runs*100:.1f}%")
    
    if aggregated_results:
        print(f"\nModel Performance (Mean ± Std):")
        for model_type, results in aggregated_results.items():
            mean_score = results['mean_score']
            std_score = results['std_score']
            print(f"  {model_type:15}: {mean_score:.4f} ± {std_score:.4f}")
    
    if failed_runs:
        print(f"\nFailed runs: {', '.join(failed_runs)}")
    
    print(f"\nResults saved:")
    print(f"  Aggregated: {agg_file}")
    if ensemble_file:
        print(f"  Ensemble: {ensemble_file}")
    print(f"  Individual fold results: cv_results/")
    print("="*80)

if __name__ == "__main__":
    main()