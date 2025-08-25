#!/usr/bin/env python3
"""
Enhanced GNN Training Runner with Optimization Results
Runs multiple GNN models using optimized hyperparameters when available.
"""

import subprocess
import sys
import os
import json
import glob
from datetime import datetime

def get_latest_optimization_results(model_type, data_file):
    """Get the latest optimization results for a specific model."""
    results_dir = "optimization_results"
    
    if not os.path.exists(results_dir):
        return None
        
    # Look for result files for this model
    pattern = os.path.join(results_dir, f"{model_type}_*{data_file}*.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        return None
        
    # Get the most recent file
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

def build_training_command(model_type, data_file, epochs=100, run_id=1):
    """Build training command with optimized parameters."""
    # Get optimization results or fallback to defaults
    opt_params = get_latest_optimization_results(model_type, data_file)
    
    if opt_params:
        print(f"✓ Using optimized parameters for {model_type} (run {run_id})")
        params = opt_params
    else:
        print(f"⚠ Using fallback parameters for {model_type} (run {run_id})")
        params = get_fallback_params(model_type)
    
    # Base command - use 'mlp' for mlp_non_network since that's what the script accepts
    actual_model_type = 'mlp' if model_type == 'mlp_non_network' else model_type
    cmd = f"python3 Enhanced_Training.py --data_file {data_file} --model_type {actual_model_type}"
    
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
    
    # Add optimized loss function parameters
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
    
    # # GraphSAINT for memory efficiency (except MLP variants)
    # if model_type not in ['mlp', 'mlp_non_network']:
    if model_type == 'gat':
        cmd += " --use_graphsaint --batch_size 2048 --walk_length 2 --num_steps 8"
    else:
        cmd += " --use_graphsaint --batch_size 8192 --walk_length 2 --num_steps 8"
    
    cmd += f" --epochs {epochs} --device auto --run_id run{run_id}"
    
    return cmd

def run_command(command):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print(f"{'='*60}")
    
    try:
        # Split command into parts, handling multi-line strings
        cmd_clean = ' '.join(command.split())
        cmd_parts = cmd_clean.split()
        result = subprocess.run(cmd_parts, check=True)
        print(f"✓ Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with return code {e.returncode}")
        return False

def main():
    """Execute the GNN training commands with optimization results."""
    
    print("Enhanced GNN Training Pipeline with Optimization Results")
    print("=" * 60)
    
    # Configuration
    data_file = "data_quantile_Target_QC_aggcat"
    non_network_data_file = "data_nonnetwork_quantile_Target_QC_aggcat"
    model_types = ['mlp_non_network', 'improved_gnn', 'residual_gcn', 'residual_sage', 'gat', 'mlp']
    epochs = 150
    
    # Show available optimization results
    results_dir = "optimization_results"
    if os.path.exists(results_dir):
        available_results = []
        for model_type in model_types:
            pattern = os.path.join(results_dir, f"{model_type}_*{data_file}*.json")
            if glob.glob(pattern):
                available_results.append(model_type)
        
        if available_results:
            print(f"✓ Found optimization results for: {', '.join(available_results)}")
        else:
            print("⚠ No optimization results found, using fallback parameters")
    else:
        print("⚠ No optimization results directory found, using fallback parameters")
    
    print(f"📋 Training {len(model_types)} models with {epochs} epochs each (3 runs per model)")
    print(f"📋 Total training runs: {len(model_types) * 3}")
    print("=" * 60)
    
    # Build and execute training commands (3 runs per model)
    successful_runs = 0
    failed_models = []
    total_runs = len(model_types) * 3
    run_counter = 0
    
    for model_type in model_types:
        for run_id in range(1, 4):  # Train 3 models for each type
            run_counter += 1
            print(f"\nStep {run_counter}: Training {model_type} - Run {run_id} ({run_counter}/{total_runs})")
            print("-" * 50)
            
            # Use non-network dataset for mlp_non_network model
            current_data_file = non_network_data_file if model_type == 'mlp_non_network' else data_file
            
            # Build command with optimized parameters
            command = build_training_command(model_type, current_data_file, epochs, run_id)
            
            # Show key parameters being used
            opt_params = get_latest_optimization_results(model_type, current_data_file)
            if not opt_params:
                opt_params = get_fallback_params(model_type)
            
            key_params = ['hidden_dim', 'num_layers', 'lr', 'dropout', 'loss_type']
            param_str = ', '.join([f"{k}={opt_params.get(k, 'N/A')}" for k in key_params if k in opt_params])
            print(f"📊 Parameters: {param_str}")
            print(f"📁 Data file: {current_data_file}")
            
            # Show loss function parameters if using class_balanced_focal
            if opt_params.get('loss_type') == 'class_balanced_focal':
                loss_params = ['cb_focal_beta', 'cb_focal_gamma', 'min_class_boost', 'oversample_strategy', 'min_samples_factor']
                loss_str = ', '.join([f"{k}={opt_params.get(k, 'N/A')}" for k in loss_params if k in opt_params])
                print(f"🎯 Loss params: {loss_str}")
            
            if model_type == 'gat' and 'heads' in opt_params:
                print(f"🎯 GAT heads: {opt_params['heads']}")
            
            # Execute training
            if run_command(command):
                successful_runs += 1
                print(f"✓ {model_type} run {run_id} training completed successfully")
            else:
                failed_models.append(f"{model_type}_run{run_id}")
                print(f"✗ {model_type} run {run_id} training failed")
    
    # Final Summary
    print("\n" + "="*60)
    print("TRAINING PIPELINE SUMMARY")
    print("="*60)
    print(f"Data file: {data_file}")
    print(f"Model types: {len(model_types)}")
    print(f"Total training runs: {total_runs}")
    print(f"Successful runs: {successful_runs}/{total_runs}")
    
    if successful_runs == total_runs:
        print("🎉 All model runs trained successfully!")
        print(f"✓ {len(model_types)} model types × 3 runs each = {total_runs} total models")
        print("\nKey Features Used:")
        print("  • Multiple runs per model type for robustness")
        print("  • Unique run identifiers for each model")
        print("  • Optimized hyperparameters (when available)")
        print("  • Class-balanced loss functions")
        print("  • GraphSAINT sampling for memory efficiency")
        print("  • Enhanced gradient handling")
    else:
        print(f"⚠ {total_runs - successful_runs} model runs failed to train")
        if failed_models:
            print(f"Failed runs: {', '.join(failed_models)}")
    
    # Show where results are saved
    print("Model checkpoints saved in: check/")
    print("Loss visualizations saved in: lossVisual/")
    print("="*60)

if __name__ == "__main__":
    main()

