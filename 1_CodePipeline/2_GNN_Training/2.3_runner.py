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
            'lr': 0.01, 'weight_decay': 1e-4, 'loss_type': 'class_balanced_focal'
        },
        'residual_gcn': {
            'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.3,
            'lr': 0.01, 'weight_decay': 1e-4, 'loss_type': 'class_balanced_focal'
        },
        'gat': {
            'hidden_dim': 64, 'num_layers': 2, 'heads': 4, 'dropout': 0.5,
            'lr': 0.005, 'weight_decay': 1e-4, 'loss_type': 'focal'
        },
        'residual_sage': {
            'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.4,
            'lr': 0.01, 'weight_decay': 1e-4, 'loss_type': 'class_balanced_focal'
        },
        'mlp': {
            'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.5,
            'lr': 0.01, 'weight_decay': 1e-3, 'loss_type': 'weighted_ce'
        }
    }
    return fallback_params.get(model_type, {})

def build_training_command(model_type, data_file, epochs=100):
    """Build training command with optimized parameters."""
    # Get optimization results or fallback to defaults
    opt_params = get_latest_optimization_results(model_type, data_file)
    
    if opt_params:
        print(f"✓ Using optimized parameters for {model_type}")
        params = opt_params
    else:
        print(f"⚠ Using fallback parameters for {model_type}")
        params = get_fallback_params(model_type)
    
    # Base command
    cmd = f"python3 Enhanced_Training.py --data_file {data_file} --model_type {model_type}"
    
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
    
    # GAT-specific parameters
    if model_type == 'gat' and 'heads' in params:
        cmd += f" --heads {params['heads']}"
    
    # GraphSAINT for memory efficiency (except MLP)
    if model_type != 'mlp':
        if model_type == 'gat':
            cmd += " --use_graphsaint --batch_size 2048 --walk_length 2 --num_steps 8"
        else:
            cmd += " --use_graphsaint --batch_size 6000 --walk_length 2 --num_steps 8"
    
    cmd += f" --epochs {epochs} --device auto"
    
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
    model_types = ['improved_gnn', 'residual_gcn', 'residual_sage', 'gat', 'mlp']
    epochs = 100
    
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
    
    print(f"📋 Training {len(model_types)} models with {epochs} epochs each")
    print("=" * 60)
    
    # Build and execute training commands
    successful_runs = 0
    failed_models = []
    
    for i, model_type in enumerate(model_types, 1):
        print(f"\nStep {i}: Training {model_type} ({i}/{len(model_types)})")
        print("-" * 40)
        
        # Build command with optimized parameters
        command = build_training_command(model_type, data_file, epochs)
        
        # Show key parameters being used
        opt_params = get_latest_optimization_results(model_type, data_file)
        if not opt_params:
            opt_params = get_fallback_params(model_type)
        
        key_params = ['hidden_dim', 'num_layers', 'lr', 'dropout', 'loss_type']
        param_str = ', '.join([f"{k}={opt_params.get(k, 'N/A')}" for k in key_params if k in opt_params])
        print(f"📊 Parameters: {param_str}")
        
        if model_type == 'gat' and 'heads' in opt_params:
            print(f"🎯 GAT heads: {opt_params['heads']}")
        
        # Execute training
        if run_command(command):
            successful_runs += 1
            print(f"✓ {model_type} training completed successfully")
        else:
            failed_models.append(model_type)
            print(f"✗ {model_type} training failed")
    
    # Final Summary
    print("\n" + "="*60)
    print("TRAINING PIPELINE SUMMARY")
    print("="*60)
    print(f"Data file: {data_file}")
    print(f"Models trained: {len(model_types)}")
    print(f"Successful runs: {successful_runs}/{len(model_types)}")
    
    if successful_runs == len(model_types):
        print("🎉 All models trained successfully!")
        print("\nKey Features Used:")
        print("  • Optimized hyperparameters (when available)")
        print("  • Class-balanced loss functions")
        print("  • GraphSAINT sampling for memory efficiency")
        print("  • Enhanced gradient handling")
    else:
        print(f"⚠ {len(model_types) - successful_runs} models failed to train")
        if failed_models:
            print(f"Failed models: {', '.join(failed_models)}")
    
    # Show where results are saved
    print(f"\nModel checkpoints saved in: check/")
    print(f"Loss visualizations saved in: lossVisual/")
    print("="*60)

if __name__ == "__main__":
    main()

