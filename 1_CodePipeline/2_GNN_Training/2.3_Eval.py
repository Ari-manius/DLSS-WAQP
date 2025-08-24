#%%
from pathlib import Path
import os

new_path = Path('/Users/ramius/Desktop/CodeVault/01_Project/Uni/DLSS_DeepLearningforSocialScientists/Final_Project/DLSS-WAQP/1_CodePipeline/2_GNN_Training')
os.chdir(new_path)

# %% 
import torch 
from utils.create_split_masks import create_split_masks
from utils.evaluate_gnn_model import evaluate_gnn_model
from utils.evaluate_gnn_model_lazy import evaluate_gnn_model_lazy
from utils.model_loader import get_model_info
from utils.GNN_model import ImprovedGNN, ResidualGCN, GraphAttentionNet, ResidualGraphSAGE, MLPBaseline

def model_data_judged_auto(data, check, use_lazy_loading=False, batch_size=1000):
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
    
    # Create model with config parameters
    model_class = model_classes[config['model_type']]
    model_kwargs = {
        'input_dim': config['input_dim'],
        'hidden_dim': config['hidden_dim'], 
        'output_dim': config['output_dim'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout']
    }
    
    if config['model_type'] == 'gat':
        model_kwargs['heads'] = config.get('heads', 4)
        
    model = model_class(**model_kwargs)
    
    # Load state dict on CPU first
    checkpoint_path = f'check/{check}.pt'.replace('.pt', '_with_config.pt') if not f'check/{check}.pt'.endswith('_with_config.pt') else f'check/{check}.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to MPS if available
    if use_lazy_loading:
        device = 'mps'
    else:
        device = 'cpu'
    
    model = model.to(device)
    model.eval()
    
    if use_lazy_loading:
        # Use lazy loading evaluation - only pass the data path
        data_path = f"data/{data}.pt"
        result = evaluate_gnn_model_lazy(data_path, model, mask_type='test', device=device, batch_size=batch_size)
    else:
        # Original evaluation method - load full graph
        data_classification = torch.load(f"data/{data}.pt", weights_only=False)
        _, _, test_mask = create_split_masks(data_classification)
        data_classification.test_mask = test_mask
        result = evaluate_gnn_model(data_classification, model, mask_type='test', device=device)
    
    return result, config


#%%
result_mlp1, config_mlp1 = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_mlp_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config_mlp1['model_type']} with hidden_dim={config_mlp1['hidden_dim']}")

#%%
result_mlp2, config_mlp2 = model_data_judged_auto("data_nonnetwork_quantile_Target_QC_aggcat", "enhanced_mlp_data_nonnetwork_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config_mlp2['model_type']} with hidden_dim={config_mlp2['hidden_dim']}")

#%%
result_residualGCN, config_residualGCN = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_residual_gcn_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config_residualGCN['model_type']} with hidden_dim={config_residualGCN['hidden_dim']}")

#%%
result_improvedGCN, config_improvedGCN = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_improved_gnn_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config_improvedGCN['model_type']} with hidden_dim={config_improvedGCN['hidden_dim']}")

#%%
result_residualSAGE, config_residualSAGE = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_residual_sage_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config_residualSAGE['model_type']} with hidden_dim={config_residualSAGE['hidden_dim']}")

#%%
result_GAT, config_GAT = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_gat_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config_GAT['model_type']} with hidden_dim={config_GAT['hidden_dim']}")

#%%
results_dict = {
    "GCN" : result_improvedGCN,
    "Residual GCN" : result_residualGCN,
    "GAT" : result_GAT,
    "Residual Sage" : result_residualSAGE,
    "MLP Full-Features" : result_mlp1,
    "MLP Article-Features" : result_mlp2,
}

#%%
import matplotlib.pyplot as plt
import numpy as np

def plot_macro_metrics_comparison(results_dict):
    """
    Create subplots for each macro metric and accuracy comparing all models
    """
    # Extract macro metrics for all models
    models = list(results_dict.keys())
    metrics = ['precision', 'recall', 'f1-score', 'accuracy']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # Extract metric values for all models
        values = []
        for model in models:
            if metric == 'accuracy':
                values.append(results_dict[model][0]['accuracy'])
            else:
                macro_avg = results_dict[model][0]['macro avg']
                values.append(macro_avg[metric])
        
        # Create bar plot
        bars = axes[i].bar(models, values, alpha=0.8)
        
        if metric == 'accuracy':
            axes[i].set_title(f'{metric.title()}')
        else:
            axes[i].set_title(f'Macro Average {metric.title()}')
            
        axes[i].set_ylabel(metric.title())
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels for better readability
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("/Users/ramius/Desktop/CodeVault/01_Project/Uni/DLSS_DeepLearningforSocialScientists/Final_Project/DLSS-WAQP/2_FinalReport/Images/Result_GNN_Metrics.png", dpi=300, bbox_inches='tight')
    #plt.show()

# Create the comparison plot
plot_macro_metrics_comparison(results_dict)

#%%
import seaborn as sns

def plot_confusion_matrices_comparison(results_dict):
    """
    Create subplots for confusion matrices of all models showing share of total
    """
    models = list(results_dict.keys())
    n_models = len(models)
    
    # Calculate subplot grid dimensions
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
    
    for i, model in enumerate(models):
        # Extract confusion matrix (second element in the tuple)
        cm = results_dict[model][1]
        
        # Convert to share of class total (row-wise normalization)
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        # Create heatmap with percentage values and uniform color scale
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                   ax=axes[i], cbar=True, vmin=0, vmax=1)
        axes[i].set_title(f'{model} Confusion Matrix (Share of Class)')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("/Users/ramius/Desktop/CodeVault/01_Project/Uni/DLSS_DeepLearningforSocialScientists/Final_Project/DLSS-WAQP/2_FinalReport/Images/Confusion_GNN_ShareOfClass.png", dpi=300, bbox_inches='tight')
    #plt.show()

# Create confusion matrix comparison plot
plot_confusion_matrices_comparison(results_dict)