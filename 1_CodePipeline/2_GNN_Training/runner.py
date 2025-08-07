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
    
    # Step 1: Data Preparation
    print("\nStep 1: Preparing data...")
    if not run_command("python3 2.1_PrepData.py"):
        print("Data preparation failed. Stopping pipeline.")
        sys.exit(1)
    
    # Step 2: Training commands
    training_commands = [
        "python3 2.2_GNN_GCN_Reg.py --data_file data_standard_Target_QC_numlog",
        "python3 2.2_GNN_Sage_Reg.py --data_file data_standard_Target_QC_numlog",
        "python3 2.2_GNN_GCN_Clas.py --data_file data_standard_Target_QC_aggcat",
        "python3 2.2_GNN_Sage_Clas.py --data_file data_standard_Target_QC_aggcat"
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