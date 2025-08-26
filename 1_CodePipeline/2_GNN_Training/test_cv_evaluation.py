#!/usr/bin/env python3
"""
Test script to verify cross-validation evaluation functionality.
"""

import torch
from torch_geometric.data import Data
from utils.create_split_masks import create_cv_splits_from_run_id, create_split_masks_regression

def test_cv_splits_consistency():
    """Test that CV splits are consistent between training and evaluation."""
    print("Testing CV splits consistency...")
    
    # Create dummy data
    x = torch.randn(1000, 10)
    edge_index = torch.randint(0, 1000, (2, 2000))
    y = torch.randint(0, 3, (1000,))
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=1000)
    
    print("\n1. Testing Random CV splits:")
    for run_id in ['run1', 'run2', 'run3']:
        # Get splits as they would be used in training
        train1, val1, test1 = create_cv_splits_from_run_id(data, run_id, n_folds=3, use_kfold=False)
        
        # Get splits as they would be used in evaluation (should be identical)
        train2, val2, test2 = create_cv_splits_from_run_id(data, run_id, n_folds=3, use_kfold=False)
        
        # Check consistency
        consistent = torch.equal(train1, train2) and torch.equal(val1, val2) and torch.equal(test1, test2)
        print(f"  {run_id}: Consistent = {consistent}, Test size = {test1.sum()}")
        
        # Verify no overlap between splits
        no_overlap = (train1 & val1).sum() == 0 and (train1 & test1).sum() == 0 and (val1 & test1).sum() == 0
        print(f"         No overlap = {no_overlap}")
    
    print("\n2. Testing K-fold CV splits:")
    for run_id in ['run1', 'run2', 'run3']:
        train1, val1, test1 = create_cv_splits_from_run_id(data, run_id, n_folds=3, use_kfold=True)
        train2, val2, test2 = create_cv_splits_from_run_id(data, run_id, n_folds=3, use_kfold=True)
        
        consistent = torch.equal(train1, train2) and torch.equal(val1, val2) and torch.equal(test1, test2)
        print(f"  {run_id}: Consistent = {consistent}, Test size = {test1.sum()}")
        
        no_overlap = (train1 & val1).sum() == 0 and (train1 & test1).sum() == 0 and (val1 & test1).sum() == 0
        print(f"         No overlap = {no_overlap}")
    
    print("\n3. Testing run_id extraction from model names:")
    test_names = [
        "enhanced_mlp_data_quantile_Target_QC_aggcat_run1",
        "enhanced_mlp_data_quantile_Target_QC_aggcat_run2", 
        "enhanced_residual_sage_data_quantile_Target_QC_aggcat_run3",
        "some_model_without_run_id"
    ]
    
    for name in test_names:
        parts = name.split('_')
        run_id = None
        for part in parts:
            if part.startswith('run') and part[3:].isdigit():
                run_id = part
                break
        print(f"  '{name}' -> run_id: {run_id}")

if __name__ == "__main__":
    test_cv_splits_consistency()
    print("\n✅ CV evaluation test completed!")