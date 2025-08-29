#%%
"""
Enhanced GNN Model Evaluation with Cross-Validation Support

This script now supports evaluating models with the same cross-validation splits 
used during training, ensuring consistent evaluation across different data partitions.

CV Options:
- USE_CV_SPLITS: Use different data splits for each run (True/False)
- USE_KFOLD: Use proper k-fold CV where each run tests on different fold (True/False) 
- n_folds: Number of folds for k-fold CV (default: 3)

The script automatically extracts run_id from model names (e.g., "model_run2" -> "run2")
and applies the same data split that was used during training.
"""

from pathlib import Path
import os

new_path = Path('/Users/ramius/Desktop/CodeVault/01_Project/Uni/DLSS_DeepLearningforSocialScientists/Final_Project/DLSS-WAQP/1_CodePipeline/2_GNN_Training')
os.chdir(new_path)

# %% 
import torch 
import matplotlib.pyplot as plt
import numpy as np
from utils.create_split_masks import create_split_masks
from utils.evaluate_gnn_model import evaluate_gnn_model
from utils.evaluate_gnn_model_lazy import evaluate_gnn_model_lazy
from utils.model_loader import get_model_info
from utils.GNN_model import ImprovedGNN, ResidualGCN, GraphAttentionNet, ResidualGraphSAGE, MLPBaseline

def model_data_judged_auto(data, check, use_lazy_loading=False, batch_size=1000, use_cv_splits=True, use_kfold=False, n_folds=3):
    # Load configuration
    config = get_model_info(f'check/{check}.pt')
    
    # Model class mapping
    model_classes = {
        'improved_gnn': ImprovedGNN,
        'residual_gcn': ResidualGCN, 
        'gat': GraphAttentionNet,
        'residual_sage': ResidualGraphSAGE, 
        'mlp': MLPBaseline,
    }
    
    # Load checkpoint to inspect actual model structure
    checkpoint_path = f'check/{check}.pt'.replace('.pt', '_with_config.pt') if not f'check/{check}.pt'.endswith('_with_config.pt') else f'check/{check}.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model with config parameters
    model_class = model_classes[config['model_type']]
    model_kwargs = {
        'input_dim': config['input_dim'],
        'hidden_dim': config['hidden_dim'], 
        'output_dim': config['output_dim'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout']
    }
    
    # For GAT, infer heads from saved model structure
    if config['model_type'] == 'gat':
        # Inspect the saved state dict to determine the number of heads
        state_dict = checkpoint['model_state_dict']
        if 'convs.0.att_src' in state_dict:
            # att_src shape is [1, heads, hidden_dim]
            saved_heads = state_dict['convs.0.att_src'].shape[1]
            model_kwargs['heads'] = saved_heads
            print(f"Detected GAT heads from checkpoint: {saved_heads}")
        else:
            model_kwargs['heads'] = config.get('heads', 4)
        
    model = model_class(**model_kwargs)
    
    # Load state dict - handle torch.compile() prefix issue
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        # Remove _orig_mod prefix from compiled model state dict
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                cleaned_key = key[10:]  # Remove '_orig_mod.' prefix
                cleaned_state_dict[cleaned_key] = value
            else:
                cleaned_state_dict[key] = value
        print("🔧 Fixed torch.compile() prefix in state dict")
        model.load_state_dict(cleaned_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    # Move to MPS if available
    if use_lazy_loading:
        device = 'mps'
    else:
        device = 'cpu'
    
    model = model.to(device)
    model.eval()
    
    if use_lazy_loading:
        # Use lazy loading evaluation - extract run_id for CV splits
        data_path = f"data/{data}.pt"
        run_id = None
        if use_cv_splits and 'run' in check:
            parts = check.split('_')
            for part in parts:
                if part.startswith('run') and part[3:].isdigit():
                    run_id = part
                    break
        
        result = evaluate_gnn_model_lazy(
            data_path, model, mask_type='test', device=device, batch_size=batch_size,
            run_id=run_id, use_cv_splits=use_cv_splits, use_kfold=use_kfold, n_folds=n_folds
        )
    else:
        # Original evaluation method - load full graph
        data_classification = torch.load(f"data/{data}.pt", weights_only=False)
        
        # Extract run_id from checkpoint name to use same data split as training
        if use_cv_splits and 'run' in check:
            # Extract run_id from model name (e.g., "enhanced_mlp_data_quantile_Target_QC_aggcat_run1" -> "run1")
            run_id = None
            parts = check.split('_')
            for part in parts:
                if part.startswith('run') and part[3:].isdigit():
                    run_id = part
                    break
            
            if run_id:
                from utils.create_split_masks import create_cv_splits_from_run_id
                _, _, test_mask = create_cv_splits_from_run_id(
                    data_classification, run_id, n_folds=n_folds, use_kfold=use_kfold
                )
                print(f"🎯 Using CV test split for {run_id} (kfold={use_kfold})")
            else:
                _, _, test_mask = create_split_masks(data_classification)
                print("⚠️  Could not extract run_id, using standard split")
        else:
            _, _, test_mask = create_split_masks(data_classification)
            print("📊 Using standard test split")
            
        data_classification.test_mask = test_mask
        result = evaluate_gnn_model(data_classification, model, mask_type='test', device=device)
    
    return result, config


# REMOVED: Single run evaluations - using cross-validation approach instead
# This eliminates redundant model loading and focuses on robust CV evaluation

#%%
def evaluate_cross_validation(model_base_names, data_name, runs=[1, 2, 3], use_lazy_loading=True, batch_size=8192, use_cv_splits=True, use_kfold=False, n_folds=3):
    """
    Evaluate multiple runs of models and calculate cross-validation statistics
    
    Args:
        model_base_names: List of base model names (without _run suffix)
        data_name: Name of the dataset
        runs: List of run numbers to evaluate
        use_lazy_loading: Whether to use lazy loading
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with CV statistics for each model
    """
    cv_results = {}
    
    for model_base in model_base_names:
        print(f"\nEvaluating {model_base}...")
        run_results = []
        
        for run in runs:
            model_name = f"{model_base}_run{run}"
            try:
                result, config = model_data_judged_auto(
                    data_name, model_name, use_lazy_loading, batch_size, 
                    use_cv_splits, use_kfold, n_folds
                )
                run_results.append(result)
                print(f"  Run {run}: Accuracy = {result[0]['accuracy']:.4f}")
            except Exception as e:
                print(f"  Run {run}: Failed - {e}")
                continue
        
        if len(run_results) > 0:
            # Calculate statistics across runs
            accuracies = [r[0]['accuracy'] for r in run_results]
            precisions = [r[0]['macro avg']['precision'] for r in run_results]
            recalls = [r[0]['macro avg']['recall'] for r in run_results]
            f1_scores = [r[0]['macro avg']['f1-score'] for r in run_results]
            
            cv_results[model_base] = {
                'n_runs': len(run_results),
                'accuracy': {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'values': accuracies
                },
                'precision': {
                    'mean': np.mean(precisions),
                    'std': np.std(precisions),
                    'values': precisions
                },
                'recall': {
                    'mean': np.mean(recalls),
                    'std': np.std(recalls),
                    'values': recalls
                },
                'f1_score': {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores),
                    'values': f1_scores
                }
            }
    
    return cv_results

#%%
# Define model base names (without _run suffix)
model_bases = [
    'enhanced_improved_gnn_data_quantile_Target_QC_aggcat',
    'enhanced_residual_gcn_data_quantile_Target_QC_aggcat', 
    'enhanced_gat_data_quantile_Target_QC_aggcat',
    'enhanced_residual_sage_data_quantile_Target_QC_aggcat',
    'enhanced_mlp_data_quantile_Target_QC_aggcat',
    'enhanced_mlp_data_nonnetwork_quantile_Target_QC_aggcat'
]

# Define model base names with their corresponding datasets
model_dataset_pairs = [
    ('enhanced_improved_gnn_data_quantile_Target_QC_aggcat', 'data_quantile_Target_QC_aggcat'),
    ('enhanced_residual_gcn_data_quantile_Target_QC_aggcat', 'data_quantile_Target_QC_aggcat'), 
    ('enhanced_gat_data_quantile_Target_QC_aggcat', 'data_quantile_Target_QC_aggcat'),
    ('enhanced_residual_sage_data_quantile_Target_QC_aggcat', 'data_quantile_Target_QC_aggcat'),
    ('enhanced_mlp_data_quantile_Target_QC_aggcat', 'data_quantile_Target_QC_aggcat'),
    ('enhanced_mlp_data_nonnetwork_quantile_Target_QC_aggcat', 'data_nonnetwork_quantile_Target_QC_aggcat')
]

# Efficient evaluation collecting both metrics and confusion matrices in one pass
cv_results = {}
confusion_matrices_by_model = {}
model_names = ['GCN', 'Residual GCN', 'GAT', 'Residual SAGE', 'MLP Full-Features', 'MLP Article-Features']

for i, (model_base, data_name) in enumerate(model_dataset_pairs):
    print(f"\nEvaluating {model_base} with {data_name}...")
    run_results = []
    confusion_matrices = []
    
    for run in [1, 2, 3]:
        model_name = f"{model_base}_run{run}"
        try:
            # CONFIGURE CV SETTINGS HERE
            USE_CV_SPLITS = True  # Set to True to use CV splits, False for standard splits
            USE_KFOLD = False     # Set to True for k-fold CV, False for random splits
            
            result, config = model_data_judged_auto(
                data_name, model_name, use_lazy_loading=True, batch_size=8192,
                use_cv_splits=USE_CV_SPLITS, use_kfold=USE_KFOLD, n_folds=3
            )
            run_results.append(result)
            confusion_matrices.append(result[1])  # Store confusion matrix
            print(f"  Run {run}: Accuracy = {result[0]['accuracy']:.4f}")
        except Exception as e:
            print(f"  Run {run}: Failed - {e}")
            continue
    
    if len(run_results) > 0:
        # Calculate statistics across runs
        accuracies = [r[0]['accuracy'] for r in run_results]
        precisions = [r[0]['macro avg']['precision'] for r in run_results]
        recalls = [r[0]['macro avg']['recall'] for r in run_results]
        f1_scores = [r[0]['macro avg']['f1-score'] for r in run_results]
        
        cv_results[model_base] = {
            'n_runs': len(run_results),
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'values': accuracies
            },
            'precision': {
                'mean': np.mean(precisions),
                'std': np.std(precisions),
                'values': precisions
            },
            'recall': {
                'mean': np.mean(recalls),
                'std': np.std(recalls),
                'values': recalls
            },
            'f1_score': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'values': f1_scores
            }
        }
        
        # Store confusion matrices for averaging
        confusion_matrices_by_model[model_names[i]] = confusion_matrices

#%%
def plot_cv_results_with_error_bars(cv_results):
    """
    Plot cross-validation results with error bars showing standard deviation
    """
    models = list(cv_results.keys())
    model_names = models  # Use actual model keys instead of hardcoded list
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        means = [cv_results[model][metric]['mean'] for model in models]
        stds = [cv_results[model][metric]['std'] for model in models]
        
        bars = axes[i].bar(model_names, means, yerr=stds, capsize=5, alpha=0.8, 
                          error_kw={'elinewidth': 2, 'capthick': 2})
        
        axes[i].set_title(f'{label} (Mean ± Std)')
        axes[i].set_ylabel(label)
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("/Users/ramius/Desktop/CodeVault/01_Project/Uni/DLSS_DeepLearningforSocialScientists/Final_Project/DLSS-WAQP/2_FinalReport/Images/CV_Results_with_ErrorBars.png", 
                dpi=300, bbox_inches='tight')
    #plt.show()

# Plot CV results
plot_cv_results_with_error_bars(cv_results)

#%%
def print_cv_summary(cv_results):
    """
    Print a summary table of cross-validation results
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    model_names = ['GCN', 'Residual GCN', 'GAT', 'Residual SAGE', 'MLP Full-Features', 'MLP Article-Features']
    
    for i, (model, name) in enumerate(zip(cv_results.keys(), model_names)):
        data = cv_results[model]
        print(f"\n{name} (n={data['n_runs']} runs):")
        print(f"  Accuracy:  {data['accuracy']['mean']:.4f} ± {data['accuracy']['std']:.4f}")
        print(f"  Precision: {data['precision']['mean']:.4f} ± {data['precision']['std']:.4f}")  
        print(f"  Recall:    {data['recall']['mean']:.4f} ± {data['recall']['std']:.4f}")
        print(f"  F1-Score:  {data['f1_score']['mean']:.4f} ± {data['f1_score']['std']:.4f}")

print_cv_summary(cv_results)

#%%
# OLD INEFFICIENT FUNCTION REMOVED - using cached data instead

# Generate averaged confusion matrices using cached data (no re-evaluation!)
def plot_averaged_confusion_matrices_efficient(confusion_matrices_by_model):
    """
    Create confusion matrices averaged across multiple runs using cached results
    """
    import seaborn as sns
    
    # Plot averaged confusion matrices
    n_models = len(confusion_matrices_by_model)
    cols = 3 if n_models > 4 else 2
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Handle case where there's only one subplot
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    averaged_cms = {}
    for i, (model_name, confusion_matrices) in enumerate(confusion_matrices_by_model.items()):
        if len(confusion_matrices) > 0:
            # Average confusion matrices across runs
            averaged_cm = np.mean(confusion_matrices, axis=0)
            averaged_cms[model_name] = averaged_cm
            
            # Convert to share of class total (row-wise normalization)
            cm_normalized = averaged_cm.astype('float') / averaged_cm.sum(axis=1, keepdims=True)
            
            # Create heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                       ax=axes[i], cbar=True, vmin=0, vmax=1)
            axes[i].set_title(f'{model_name}\nConfusion Matrix (Avg across {len(confusion_matrices)} runs)')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for j in range(len(confusion_matrices_by_model), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("/Users/ramius/Desktop/CodeVault/01_Project/Uni/DLSS_DeepLearningforSocialScientists/Final_Project/DLSS-WAQP/2_FinalReport/Images/Averaged_Confusion_Matrices.png", 
                dpi=300, bbox_inches='tight')
    #plt.show()
    
    return averaged_cms

# Generate averaged confusion matrices efficiently (no model re-loading!)
averaged_confusion_matrices = plot_averaged_confusion_matrices_efficient(confusion_matrices_by_model)

#%%
# Save CV results as JSON
import json

# Convert numpy arrays to lists for JSON compatibility
cv_results_json = {}
for model, data in cv_results.items():
    cv_results_json[model] = {
        'n_runs': data['n_runs'],
        'accuracy': {
            'mean': float(data['accuracy']['mean']),
            'std': float(data['accuracy']['std']),
            'values': [float(x) for x in data['accuracy']['values']]
        },
        'precision': {
            'mean': float(data['precision']['mean']),
            'std': float(data['precision']['std']),
            'values': [float(x) for x in data['precision']['values']]
        },
        'recall': {
            'mean': float(data['recall']['mean']),
            'std': float(data['recall']['std']),
            'values': [float(x) for x in data['recall']['values']]
        },
        'f1_score': {
            'mean': float(data['f1_score']['mean']),
            'std': float(data['f1_score']['std']),
            'values': [float(x) for x in data['f1_score']['values']]
        }
    }

with open('cv_results.json', 'w') as f:
    json.dump(cv_results_json, f, indent=2)

print("CV results saved to cv_results.json")