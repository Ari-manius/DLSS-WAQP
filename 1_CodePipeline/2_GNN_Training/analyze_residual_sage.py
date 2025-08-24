#!/usr/bin/env python3
"""
Focused script to analyze Residual SAGE model performance and understand
the precision calculation discrepancy.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Set the working directory
new_path = Path('/Users/ramius/Desktop/CodeVault/01_Project/Uni/DLSS_DeepLearningforSocialScientists/Final_Project/DLSS-WAQP/1_CodePipeline/2_GNN_Training')
os.chdir(new_path)

# Import utilities
from utils.create_split_masks import create_split_masks
from utils.evaluate_gnn_model_lazy import evaluate_gnn_model_lazy
from utils.model_loader import get_model_info
from utils.GNN_model import ResidualGraphSAGE

def load_and_evaluate_residual_sage():
    """Load and evaluate the Residual SAGE model with detailed output."""
    
    print("="*80)
    print("RESIDUAL SAGE MODEL ANALYSIS")
    print("="*80)
    
    # Load model configuration
    check_name = "enhanced_residual_sage_data_quantile_Target_QC_aggcat"
    config = get_model_info(f'check/{check_name}.pt')
    
    print(f"Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create model
    model = ResidualGraphSAGE(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'], 
        output_dim=config['output_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Load model weights
    checkpoint_path = f'check/{check_name}_with_config.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    device = 'mps'
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Device: {device}")
    print()
    
    # Evaluate model
    data_path = "data/data_quantile_Target_QC_aggcat.pt"
    print("Starting evaluation...")
    print("-" * 50)
    
    # Run evaluation with detailed output
    result, confusion_mat = evaluate_gnn_model_lazy(
        data_path, model, mask_type='test', device=device, batch_size=8192
    )
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # Print detailed confusion matrix analysis
    print("\nCONFUSION MATRIX:")
    print(confusion_mat)
    
    # Analyze confusion matrix
    print("\nCONFUSION MATRIX ANALYSIS:")
    for i in range(confusion_mat.shape[0]):
        total_actual = confusion_mat[i, :].sum()
        correct = confusion_mat[i, i]
        print(f"Class {i}: {total_actual} actual samples, {correct} correctly predicted ({correct/total_actual:.3f} recall)")
    
    print("\nPer-column (predicted) analysis:")
    for j in range(confusion_mat.shape[1]):
        total_predicted = confusion_mat[:, j].sum()
        correct = confusion_mat[j, j]
        if total_predicted > 0:
            precision = correct / total_predicted
            print(f"Class {j}: {total_predicted} predicted, {correct} correct ({precision:.3f} precision)")
        else:
            print(f"Class {j}: 0 predicted samples (precision undefined/0.0)")
    
    # Print classification report details
    print("\nCLASSIFICATION REPORT BREAKDOWN:")
    print("-" * 50)
    
    for class_id in ['0', '1', '2']:
        if class_id in result:
            metrics = result[class_id]
            print(f"Class {class_id}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-score:  {metrics['f1-score']:.4f}")
            print(f"  Support:   {metrics['support']}")
            print()
    
    # Macro averages
    macro_avg = result['macro avg']
    print(f"MACRO AVERAGES:")
    print(f"  Precision: {macro_avg['precision']:.4f}")
    print(f"  Recall:    {macro_avg['recall']:.4f}")
    print(f"  F1-score:  {macro_avg['f1-score']:.4f}")
    print()
    
    # Manual calculation of macro precision
    print("MANUAL MACRO PRECISION CALCULATION:")
    class_precisions = []
    for class_id in ['0', '1', '2']:
        if class_id in result:
            precision = result[class_id]['precision']
            class_precisions.append(precision)
            print(f"  Class {class_id} precision: {precision:.4f}")
    
    manual_macro_precision = np.mean(class_precisions)
    print(f"  Manual macro precision: {manual_macro_precision:.4f}")
    print(f"  sklearn macro precision: {macro_avg['precision']:.4f}")
    print(f"  Match: {'Yes' if abs(manual_macro_precision - macro_avg['precision']) < 1e-6 else 'No'}")
    
    return result, confusion_mat

if __name__ == "__main__":
    try:
        result, confusion_mat = load_and_evaluate_residual_sage()
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()