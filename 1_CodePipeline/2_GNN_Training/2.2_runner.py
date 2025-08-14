#!/usr/bin/env python3
"""
GNN Training Pipeline Runner
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print(f"{'='*60}")
    
    try:
        # Split command into parts
        cmd_parts = command.split()
        result = subprocess.run(cmd_parts, check=True)
        print(f"✓ {command} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command} failed with return code {e.returncode}")
        return False

def main():
    """Execute the GNN training commands."""
    
    print("Starting GNN Training Pipeline")
    print("=" * 60)
    
    # # Step 1: Data Preparation
    # print("\nStep 1: Preparing data...")
    # if not run_command("python3 2.1_PrepData.py"):
    #     print("Data preparation failed. Stopping pipeline.")
    #     sys.exit(1)
    
    # Step 2: Training commands
    training_commands = [
    "python3 2.2_Enhanced_Training.py \
        --data_file data_quantile_Target_QC_aggcat \
        --model_type improved_gnn \
        --loss_type class_balanced_focal \
        --use_graphsaint \
        --batch_size 8192 \
        --walk_length 2 \
        --num_steps 8 \
        --epochs 100",
    "python3 2.2_Enhanced_Training.py \
        --data_file data_quantile_Target_QC_aggcat \
        --model_type residual_gcn \
        --loss_type class_balanced_focal \
        --use_graphsaint \
        --batch_size 8192 \
        --walk_length 2 \
        --num_steps 8 \
        --epochs 100",
    "python3 2.2_Enhanced_Training.py \
        --data_file data_quantile_Target_QC_aggcat \
        --model_type residual_sage \
        --loss_type class_balanced_focal \
        --use_graphsaint \
        --batch_size 8192 \
        --walk_length 2 \
        --num_steps 8 \
        --epochs 100",
    "python3 2.2_Enhanced_Training.py \
        --data_file data_quantile_Target_QC_aggcat \
        --model_type gat \
        --loss_type class_balanced_focal \
        --use_graphsaint \
        --batch_size 2048 \
        --walk_length 2 \
        --num_steps 8 \
        --epochs 100",
    "python3 2.2_Enhanced_Training.py \
        --data_file data_quantile_Target_QC_aggcat \
        --model_type mlp \
        --epochs 100",    
    ]
    
    successful_runs = 0
    total_runs = len(training_commands)
    
    for i, command in enumerate(training_commands, 1):
        print(f"\nStep {i+1}: Training model {i}/{total_runs}")
        if run_command(command):
            successful_runs += 1
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Successful training runs: {successful_runs}/{total_runs}")
    
    if successful_runs == total_runs:
        print("✓ All training completed successfully!")
    else:
        print(f"✗ {total_runs - successful_runs} training runs failed")
    
    print("="*60)

if __name__ == "__main__":
    main()

