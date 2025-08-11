#!/usr/bin/env python3
"""
Enhanced GNN Training Pipeline with Hyperparameter Optimization
Integrates class imbalance handling with automated hyperparameter tuning.
"""

import subprocess
import sys
import os
import json
import argparse
from datetime import datetime

def run_command(command, description=None):
    """Run a command and handle errors."""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")
    
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command.split(), check=True, capture_output=True, text=True)
        print(f"✓ Command completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False, None

def run_hyperparameter_optimization(data_file, model_types, n_trials=50, n_splits=3):
    """
    Run hyperparameter optimization for each model type.
    
    Args:
        data_file: Base name of the data file
        model_types: List of model types to optimize
        n_trials: Number of optimization trials per model
        n_splits: Number of CV folds
    
    Returns:
        Dictionary of best parameters for each model type
    """
    print("Starting Hyperparameter Optimization Phase")
    print("=" * 60)
    
    best_params_by_model = {}
    successful_optimizations = 0
    
    for model_type in model_types:
        print(f"\nOptimizing {model_type}...")
        
        optimization_command = (
            f"python3 2.3_Hyperparameter_Optimization.py "
            f"--data_file {data_file} "
            f"--model_type {model_type} "
            f"--n_trials {n_trials} "
            f"--n_splits {n_splits} "
            f"--epochs_per_trial 10 "
            f"--final_epochs 20 "
            f"--device auto"
        )
        
        success, output = run_command(
            optimization_command, 
            f"Hyperparameter optimization for {model_type}"
        )
        
        if success:
            successful_optimizations += 1
            
            # Look for the optimization results file
            results_dir = "optimization_results"
            if os.path.exists(results_dir):
                # Find the most recent results file for this model
                files = [f for f in os.listdir(results_dir) 
                        if f.startswith(f"{model_type}_{data_file}") and f.endswith('.json')]
                
                if files:
                    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
                    results_path = os.path.join(results_dir, latest_file)
                    
                    try:
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                            best_params_by_model[model_type] = results['best_params']
                            print(f"✓ Best parameters saved for {model_type}")
                    except Exception as e:
                        print(f"⚠ Could not load optimization results for {model_type}: {e}")
        else:
            print(f"✗ Optimization failed for {model_type}")
    
    print(f"\nOptimization Summary: {successful_optimizations}/{len(model_types)} models optimized")
    return best_params_by_model

def run_optimized_training(data_file, model_types, best_params_by_model, use_graphsaint=True):
    """
    Run training with optimized hyperparameters.
    
    Args:
        data_file: Base name of the data file
        model_types: List of model types to train
        best_params_by_model: Dictionary of optimized parameters
        use_graphsaint: Whether to use GraphSAINT sampling
    
    Returns:
        Number of successful training runs
    """
    print("\nStarting Optimized Training Phase")
    print("=" * 60)
    
    successful_runs = 0
    training_commands = []
    
    for model_type in model_types:
        if model_type in best_params_by_model:
            params = best_params_by_model[model_type]
            
            # Build training command with optimized parameters
            base_command = (
                f"python3 2.2_Enhanced_Training.py "
                f"--data_file {data_file} "
                f"--model_type {model_type} "
                f"--hidden_dim {params.get('hidden_dim', 128)} "
                f"--num_layers {params.get('num_layers', 3)} "
                f"--dropout {params.get('dropout', 0.4)} "
                f"--lr {params.get('lr', 0.01)} "
                f"--epochs 10"
            )
            
            # Add loss function parameters
            loss_type = params.get('loss_type', 'class_balanced_focal')
            base_command += f" --loss_type {loss_type}"
            
            # Add GraphSAINT and device parameters if enabled
            if use_graphsaint:
                # Use memory-efficient batch sizes based on model type
                if model_type == 'gat':
                    batch_size = 4000  # GAT is more memory intensive
                else:
                    batch_size = 6000  # Other models can handle larger batches
                base_command += f" --use_graphsaint --batch_size {batch_size} --walk_length 2 --num_steps 8 --memory_efficient"
            
            # Always add device parameter for automatic selection
            base_command += " --device auto"
            
            training_commands.append((model_type, base_command))
        else:
            # Fallback to default parameters if optimization failed
            print(f"⚠ Using default parameters for {model_type} (optimization failed)")
            
            fallback_command = (
                f"python3 2.2_Enhanced_Training.py "
                f"--data_file {data_file} "
                f"--model_type {model_type} "
                f"--loss_type class_balanced_focal "
                f"--epochs 10"
            )
            
            if use_graphsaint:
                # Use memory-efficient batch sizes for fallback too
                fallback_batch_size = 4000 if model_type == 'gat' else 6000
                fallback_command += f" --use_graphsaint --batch_size {fallback_batch_size} --walk_length 2 --num_steps 8 --memory_efficient"
            
            # Always add device parameter for automatic selection
            fallback_command += " --device auto"
                
            training_commands.append((model_type, fallback_command))
    
    # Execute training commands
    for i, (model_type, command) in enumerate(training_commands, 1):
        print(f"\nTraining optimized {model_type} model ({i}/{len(training_commands)})")
        
        success, _ = run_command(command, f"Training {model_type} with optimized parameters")
        
        if success:
            successful_runs += 1
        
    return successful_runs

def main():
    parser = argparse.ArgumentParser(description="Enhanced GNN training with hyperparameter optimization")
    parser.add_argument('--data_file', type=str, 
                       default='data_quantile_Target_QC_aggcat',
                       help='Base name of the .pt file (without extension)')
    parser.add_argument('--model_types', type=str, nargs='+',
                       default=['improved_gnn', 'residual_gcn', 'residual_sage', 'mlp'],
                       choices=['improved_gnn', 'residual_gcn', 'gat', 'residual_sage', 'mlp'],
                       help='Model types to optimize and train')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials per model')
    parser.add_argument('--n_splits', type=int, default=3,
                       help='Number of CV folds for optimization')
    parser.add_argument('--skip_optimization', action='store_true',
                       help='Skip hyperparameter optimization and use default enhanced parameters')
    parser.add_argument('--use_graphsaint', action='store_true', default=True,
                       help='Use GraphSAINT sampling for training')
    
    args = parser.parse_args()
    
    print("Enhanced GNN Training Pipeline with Class Imbalance Handling")
    print("=" * 60)
    print(f"Data file: {args.data_file}")
    print(f"Model types: {args.model_types}")
    print(f"Use GraphSAINT: {args.use_graphsaint}")
    
    if args.skip_optimization:
        print("⚠ Skipping hyperparameter optimization - using enhanced defaults")
        best_params_by_model = {}
    else:
        print(f"Optimization trials per model: {args.n_trials}")
        print(f"CV folds: {args.n_splits}")
        
        # Phase 1: Hyperparameter Optimization
        best_params_by_model = run_hyperparameter_optimization(
            args.data_file, args.model_types, args.n_trials, args.n_splits
        )
    
    # Phase 2: Training with Optimized Parameters
    successful_training_runs = run_optimized_training(
        args.data_file, args.model_types, best_params_by_model, args.use_graphsaint
    )
    
    # Final Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    if not args.skip_optimization:
        print(f"Models optimized: {len(best_params_by_model)}/{len(args.model_types)}")
        
        # Display best parameters
        if best_params_by_model:
            print("\nBest Parameters Found:")
            for model_type, params in best_params_by_model.items():
                print(f"\n{model_type}:")
                for param, value in params.items():
                    print(f"  {param}: {value}")
    
    print(f"\nSuccessful training runs: {successful_training_runs}/{len(args.model_types)}")
    
    if successful_training_runs == len(args.model_types):
        print("✓ All models trained successfully!")
        print("\nKey Features Used:")
        print("  • Class-Balanced Focal Loss with minority class boost")
        print("  • Smart oversampling strategy")
        print("  • Hyperparameter optimization focused on minority class F1")
        print("  • Enhanced gradient clipping and early stopping")
    else:
        failed_runs = len(args.model_types) - successful_training_runs
        print(f"✗ {failed_runs} training runs failed")
    
    print("="*60)

if __name__ == "__main__":
    main()