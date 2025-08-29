import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_all_results():
    """Load results from  GNNs, MLPs, ORES, and Random Forest"""
    
    # Load GNNs and MLPs
    with open('cv_results.json', 'r') as f:
        gnn_results = json.load(f)
    
    # Load ORES
    with open('ores_benchmark_results.json', 'r') as f:
        ores_results = json.load(f)
    
    # Load Random Forest
    with open('random_forest_results.json', 'r') as f:
        rf_results = json.load(f)
    
    return gnn_results, ores_results, rf_results

def calculate_ores_confusion_matrix():
    """Calculate ORES confusion matrix from detailed predictions"""
    
    # Load ORES detailed predictions
    ores_df = pd.read_csv('ores_detailed_predictions.csv')
    
    # Extract true and predicted labels
    y_true = ores_df['true_label_numeric'].values
    y_pred = ores_df['ores_aggcat'].values
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    return cm

def convert_ores_to_cv_format(ores_results):
    """Convert ORES results to CV format (it isnt tho (╥﹏╥) )"""
    return {
        'ores': {
            'n_runs': 1,
            'accuracy': {
                'mean': ores_results['metrics']['accuracy'],
                'std': 0.0,
                'values': [ores_results['metrics']['accuracy']]
            },
            'precision': {
                'mean': ores_results['metrics']['precision_macro'],
                'std': 0.0,
                'values': [ores_results['metrics']['precision_macro']]
            },
            'recall': {
                'mean': ores_results['metrics']['recall_macro'],
                'std': 0.0,
                'values': [ores_results['metrics']['recall_macro']]
            },
            'f1_score': {
                'mean': ores_results['metrics']['f1_macro'],
                'std': 0.0,
                'values': [ores_results['metrics']['f1_macro']]
            }
        }
    }

def create_unified_results(gnn_results, ores_results, rf_results=None):
    """Combine all results into one format"""
    
    # Convert ORES results to CV format
    ores_cv = convert_ores_to_cv_format(ores_results)

    # add gnns and mlps
    unified = gnn_results.copy()
    
    # Add ORES
    unified.update(ores_cv)
    
    # Add RF
    unified.update(rf_results)
    
    return unified

def get_model_display_names():
    """clean names for models"""
    model_mapping = {
        'enhanced_improved_gnn_data_quantile_Target_QC_aggcat': 'GCN',
        'enhanced_residual_gcn_data_quantile_Target_QC_aggcat': 'Residual GCN',
        'enhanced_gat_data_quantile_Target_QC_aggcat': 'GAT',
        'enhanced_residual_sage_data_quantile_Target_QC_aggcat': 'Residual SAGE',
        'enhanced_mlp_data_quantile_Target_QC_aggcat': 'MLP Full-Features',
        'enhanced_mlp_data_nonnetwork_quantile_Target_QC_aggcat': 'MLP Article-Features',
        'ores': 'ORES',
        'random_forest': 'Random Forest'
    }
    return model_mapping

def plot_unified_comparison(unified_results, output_dir='../../2_FinalReport/Images'):
    """Comparison plots of all methods"""

    Path(output_dir).mkdir(exist_ok=True)
    
    model_mapping = get_model_display_names()

    available_models = []
    display_names = []
    for key in unified_results.keys():
        if key in model_mapping:
            available_models.append(key)
            display_names.append(model_mapping[key])
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']
    
    # Create main plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define colors for different model types
    colors = {
        'GCN': '#1f77b4',
        'Residual GCN': '#ff7f0e', 
        'GAT': '#2ca02c',
        'Residual SAGE': '#d62728',
        'MLP Full-Features': '#9467bd',
        'MLP Article-Features': '#8c564b',
        'ORES': '#e377c2',
        'Random Forest': '#7f7f7f'
    }
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        means = [unified_results[model][metric]['mean'] for model in available_models]
        stds = [unified_results[model][metric]['std'] for model in available_models]
        
        # Use colors based on display names
        bar_colors = [colors.get(name, '#bcbd22') for name in display_names]
        
        bars = axes[i].bar(display_names, means, yerr=stds, capsize=5, alpha=0.8, 
                          color=bar_colors,
                          error_kw={'elinewidth': 2, 'capthick': 2})
        
        axes[i].set_title(f'{label} (Mean ± Std)', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(label, fontsize=12)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_plot.png', dpi=300, bbox_inches='tight')
    
    # Create accuracy comparison
    plt.figure(figsize=(12, 8))
    means = [unified_results[model]['accuracy']['mean'] for model in available_models]
    stds = [unified_results[model]['accuracy']['std'] for model in available_models]
    
    bar_colors = [colors.get(name, '#bcbd22') for name in display_names]
    bars = plt.bar(display_names, means, yerr=stds, capsize=5, alpha=0.8, 
                   color=bar_colors, error_kw={'elinewidth': 2, 'capthick': 2})
    
    plt.title('Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison_final.png', dpi=300, bbox_inches='tight')
    
    #Summary table
    create_results_table(unified_results, available_models, display_names, output_dir)
    
    plt.show()

def create_results_table(unified_results, available_models, display_names, output_dir):
    """Create results table"""
    
    # Create df
    table_data = []
    for i, model in enumerate(available_models):
        data = unified_results[model]
        table_data.append({
            'Model': display_names[i],
            'N_Runs': data['n_runs'],
            'Accuracy': f"{data['accuracy']['mean']:.4f} ± {data['accuracy']['std']:.4f}",
            'Precision': f"{data['precision']['mean']:.4f} ± {data['precision']['std']:.4f}",
            'Recall': f"{data['recall']['mean']:.4f} ± {data['recall']['std']:.4f}",
            'F1-Score': f"{data['f1_score']['mean']:.4f} ± {data['f1_score']['std']:.4f}",
            'Accuracy_Mean': data['accuracy']['mean'],  # For sorting
        })
    
    df_results = pd.DataFrame(table_data)
    
    # Sort by accuracy (descending)
    df_results = df_results.sort_values('Accuracy_Mean', ascending=False)
    df_results = df_results.drop('Accuracy_Mean', axis=1)
    
    # Save as CSV in 2_GNN_Training directory
    df_results.to_csv('model_table.csv', index=False)
    
    # Print table
    print("Final Comparison")
    print(df_results.to_string(index=False))
    
    return df_results

def plot_all_confusion_matrices(gnn_confusion_matrices, output_dir='../../2_FinalReport/Images'):
    """Create confusion matrices for all models"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load RF
    with open('random_forest_results.json', 'r') as f:
        rf_data = json.load(f)
    rf_confusion_matrices = rf_data['confusion_matrices']
    
    # Calculate ORES confusion matrix
    ores_cm = calculate_ores_confusion_matrix()
    
    # Combine
    all_confusion_matrices = {}
    
    # Add GNN/MLP confusion matrices
    model_names_gnn = ['GCN', 'Residual GCN', 'GAT', 'Residual SAGE', 'MLP Full-Features', 'MLP Article-Features']
    for i, (model_name, cms) in enumerate(gnn_confusion_matrices.items()):
        if len(cms) > 0:
            averaged_cm = np.mean(cms, axis=0)
            all_confusion_matrices[model_name] = averaged_cm
    
    # Add Random Forest
    rf_averaged = np.mean(rf_confusion_matrices, axis=0)
    all_confusion_matrices['Random Forest'] = rf_averaged
    
    # Add ORES
    all_confusion_matrices['ORES'] = ores_cm
    
    # Create the plot
    n_models = len(all_confusion_matrices)
    cols = 4 if n_models > 6 else 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Handle differentconfigs
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Class labels
    class_labels = ['Low Quality', 'Medium Quality', 'High Quality']
    
    for i, (model_name, cm) in enumerate(all_confusion_matrices.items()):
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                   ax=axes[i], cbar=True, vmin=0, vmax=1,
                   xticklabels=class_labels, yticklabels=class_labels)
        
        # Add run information to title
        if model_name == 'ORES':
            title = f'{model_name}\nConfusion Matrix'
        elif model_name == 'Random Forest':
            title = f'{model_name}\nConfusion Matrix (Avg across 3 runs)'
        else:
            n_runs = len(gnn_confusion_matrices.get(model_name, []))
            title = f'{model_name}\nConfusion Matrix (Avg across {n_runs} runs)'
            
        axes[i].set_title(title, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for j in range(len(all_confusion_matrices), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/All_Confusion_Matrices_with_RF_ORES.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_confusion_matrices

def print_summary_statistics(unified_results):
    """Print summary statistics"""
    
    model_mapping = get_model_display_names()

    for model_key, data in unified_results.items():
        if model_key in model_mapping:
            name = model_mapping[model_key]
            print(f"\n{name} (n={data['n_runs']} runs):")
            print(f"Accuracy:{data['accuracy']['mean']:.4f} ± {data['accuracy']['std']:.4f}")
            print(f"Precision: {data['precision']['mean']:.4f} ± {data['precision']['std']:.4f}")
            print(f"Recall:{data['recall']['mean']:.4f} ± {data['recall']['std']:.4f}")
            print(f"F1-Score:{data['f1_score']['mean']:.4f} ± {data['f1_score']['std']:.4f}")

def load_gnn_confusion_matrices():
    """Load confusion matrices from eval.py"""
    # This would normally be loaded from the eval.py execution
    return {}

if __name__ == "__main__":
    
    # Load all results
    gnn_results, ores_results, rf_results = load_all_results()
    
    # Create unified results
    unified_results = create_unified_results(gnn_results, ores_results, rf_results)
    
    # Print summary
    print_summary_statistics(unified_results)
    
    # Create visualizations
    plot_unified_comparison(unified_results)
    
    # Create comprehensive confusion matrices if GNN confusion matrices are available
    try:
        # Try to load confusion matrices from eval.py execution
        # If confusion_matrices_by_model exists in scope, use it
        if 'confusion_matrices_by_model' in globals():
            print("\nGenerating comprehensive confusion matrices...")
            all_cms = plot_all_confusion_matrices(confusion_matrices_by_model)
            print("Confusion matrices saved successfully!")
        else:
            print("\nGNN confusion matrices not available. Run eval.py first to generate comprehensive confusion matrices.")
    except Exception as e:
        print(f"\nCould not generate comprehensive confusion matrices: {e}")
    
    # Save combined results
    with open('unified_results.json', 'w') as f:
        json.dump(unified_results, f, indent=2)
    
    print("All results saved")