#!/usr/bin/env python3
"""
Cross-Validation Results Analyzer
Comprehensive analysis and reporting of cross-validation results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from collections import defaultdict

class CVResultsAnalyzer:
    """Analyzer for cross-validation results."""
    
    def __init__(self, results_dir="cv_results"):
        self.results_dir = results_dir
        self.results_data = defaultdict(list)
        self.model_types = []
        self.data_file = None
        
    def load_cv_results(self, data_file, k_folds=5):
        """Load all cross-validation results."""
        self.data_file = data_file
        
        print(f"Loading CV results for {data_file}...")
        
        for model_type in ['mlp_non_network', 'improved_gnn', 'residual_gcn', 'residual_sage', 'gat', 'mlp']:
            model_results = []
            
            for fold in range(k_folds):
                result_file = os.path.join(
                    self.results_dir,
                    f"{model_type}_{data_file}_fold_{fold}_results.json"
                )
                
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                        model_results.append(result)
            
            if model_results:
                self.results_data[model_type] = model_results
                self.model_types.append(model_type)
                print(f"  {model_type}: {len(model_results)} folds loaded")
            else:
                print(f"  {model_type}: No results found")
    
    def create_summary_statistics(self):
        """Create summary statistics for all models."""
        summary_stats = {}
        
        for model_type, results in self.results_data.items():
            if not results:
                continue
                
            # Extract test accuracies
            test_accs = [r['test_accuracy'] for r in results]
            val_accs = [r['best_val_accuracy'] for r in results]
            
            # Calculate statistics
            stats_dict = {
                'test_accuracy_mean': np.mean(test_accs),
                'test_accuracy_std': np.std(test_accs),
                'test_accuracy_min': np.min(test_accs),
                'test_accuracy_max': np.max(test_accs),
                'val_accuracy_mean': np.mean(val_accs),
                'val_accuracy_std': np.std(val_accs),
                'fold_count': len(test_accs),
                'test_accuracies': test_accs,
                'val_accuracies': val_accs
            }
            
            # Calculate confidence interval
            if len(test_accs) > 1:
                ci = stats.t.interval(0.95, len(test_accs)-1, 
                                     loc=np.mean(test_accs), 
                                     scale=stats.sem(test_accs))
                stats_dict['test_accuracy_ci_lower'] = ci[0]
                stats_dict['test_accuracy_ci_upper'] = ci[1]
            else:
                stats_dict['test_accuracy_ci_lower'] = stats_dict['test_accuracy_mean']
                stats_dict['test_accuracy_ci_upper'] = stats_dict['test_accuracy_mean']
            
            summary_stats[model_type] = stats_dict
        
        return summary_stats
    
    def create_detailed_comparison(self):
        """Create detailed model comparison table."""
        summary_stats = self.create_summary_statistics()
        
        comparison_data = []
        for model_type, stats in summary_stats.items():
            comparison_data.append({
                'Model': model_type,
                'Mean Accuracy': f"{stats['test_accuracy_mean']:.4f}",
                'Std Dev': f"{stats['test_accuracy_std']:.4f}",
                'Min': f"{stats['test_accuracy_min']:.4f}",
                'Max': f"{stats['test_accuracy_max']:.4f}",
                'CI Lower': f"{stats['test_accuracy_ci_lower']:.4f}",
                'CI Upper': f"{stats['test_accuracy_ci_upper']:.4f}",
                'Folds': stats['fold_count']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Mean Accuracy', ascending=False)
        
        return df
    
    def plot_accuracy_distribution(self, save_path=None):
        """Plot accuracy distribution across folds for each model."""
        summary_stats = self.create_summary_statistics()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot
        test_data = []
        labels = []
        for model_type, stats in summary_stats.items():
            test_data.extend(stats['test_accuracies'])
            labels.extend([model_type] * len(stats['test_accuracies']))
        
        df = pd.DataFrame({'Model': labels, 'Test Accuracy': test_data})
        
        sns.boxplot(data=df, x='Model', y='Test Accuracy', ax=ax1)
        ax1.set_title('Test Accuracy Distribution by Model')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Mean accuracy with error bars
        models = list(summary_stats.keys())
        means = [summary_stats[m]['test_accuracy_mean'] for m in models]
        stds = [summary_stats[m]['test_accuracy_std'] for m in models]
        
        bars = ax2.bar(models, means, yerr=stds, capsize=5, 
                      color='lightblue', edgecolor='darkblue', alpha=0.7)
        ax2.set_title('Mean Test Accuracy with Standard Deviation')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, save_path=None):
        """Plot average learning curves for all models."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (model_type, results) in enumerate(self.results_data.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Collect training histories
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            
            for result in results:
                history = result.get('training_history', {})
                if history:
                    train_losses.append(history.get('train_losses', []))
                    val_losses.append(history.get('val_losses', []))
                    train_accs.append(history.get('train_accuracies', []))
                    val_accs.append(history.get('val_accuracies', []))
            
            if train_losses and all(len(tl) > 0 for tl in train_losses):
                # Find minimum length to align all curves
                min_length = min(len(tl) for tl in train_losses)
                
                # Truncate all curves to same length
                train_losses = [tl[:min_length] for tl in train_losses]
                val_losses = [vl[:min_length] for vl in val_losses]
                train_accs = [ta[:min_length] for ta in train_accs]
                val_accs = [va[:min_length] for va in val_accs]
                
                # Calculate means and stds
                mean_train_loss = np.mean(train_losses, axis=0)
                std_train_loss = np.std(train_losses, axis=0)
                mean_val_loss = np.mean(val_losses, axis=0)
                std_val_loss = np.std(val_losses, axis=0)
                
                epochs = range(1, min_length + 1)
                
                # Plot losses
                ax.plot(epochs, mean_train_loss, label='Train Loss', color='blue')
                ax.fill_between(epochs, 
                               mean_train_loss - std_train_loss,
                               mean_train_loss + std_train_loss,
                               alpha=0.3, color='blue')
                
                ax.plot(epochs, mean_val_loss, label='Val Loss', color='red')
                ax.fill_between(epochs,
                               mean_val_loss - std_val_loss, 
                               mean_val_loss + std_val_loss,
                               alpha=0.3, color='red')
                
                ax.set_title(f'{model_type} Learning Curves')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No training history\nfor {model_type}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_type} Learning Curves')
        
        # Hide unused subplots
        for idx in range(len(self.results_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves plot saved to {save_path}")
        
        plt.show()
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests between models."""
        summary_stats = self.create_summary_statistics()
        model_names = list(summary_stats.keys())
        
        if len(model_names) < 2:
            print("Need at least 2 models for statistical comparison")
            return None
        
        # Pairwise t-tests
        comparison_results = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                acc1 = summary_stats[model1]['test_accuracies']
                acc2 = summary_stats[model2]['test_accuracies']
                
                if len(acc1) > 1 and len(acc2) > 1:
                    # Perform paired t-test if same number of folds
                    if len(acc1) == len(acc2):
                        t_stat, p_value = stats.ttest_rel(acc1, acc2)
                        test_type = "Paired t-test"
                    else:
                        t_stat, p_value = stats.ttest_ind(acc1, acc2)
                        test_type = "Independent t-test"
                    
                    comparison_results[f"{model1} vs {model2}"] = {
                        'test_type': test_type,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'mean_diff': np.mean(acc1) - np.mean(acc2)
                    }
        
        return comparison_results
    
    def generate_report(self, save_path=None):
        """Generate comprehensive CV results report."""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"cv_analysis_report_{timestamp}.txt"
        
        summary_stats = self.create_summary_statistics()
        comparison_df = self.create_detailed_comparison()
        statistical_tests = self.perform_statistical_tests()
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("CROSS-VALIDATION RESULTS ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Data File: {self.data_file}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model comparison table
        report_lines.append("MODEL PERFORMANCE COMPARISON")
        report_lines.append("-"*50)
        report_lines.append(comparison_df.to_string(index=False))
        report_lines.append("")
        
        # Best performing model
        best_model = comparison_df.iloc[0]['Model']
        best_acc = float(comparison_df.iloc[0]['Mean Accuracy'])
        report_lines.append(f"BEST PERFORMING MODEL: {best_model}")
        report_lines.append(f"Best Mean Accuracy: {best_acc:.4f}")
        report_lines.append("")
        
        # Statistical significance tests
        if statistical_tests:
            report_lines.append("STATISTICAL SIGNIFICANCE TESTS")
            report_lines.append("-"*40)
            for comparison, result in statistical_tests.items():
                sig_marker = "***" if result['significant'] else ""
                report_lines.append(f"{comparison}:")
                report_lines.append(f"  {result['test_type']}")
                report_lines.append(f"  t-statistic: {result['t_statistic']:.4f}")
                report_lines.append(f"  p-value: {result['p_value']:.4f} {sig_marker}")
                report_lines.append(f"  Mean difference: {result['mean_diff']:.4f}")
                report_lines.append("")
        
        # Model rankings
        report_lines.append("MODEL RANKINGS (by mean accuracy)")
        report_lines.append("-"*35)
        for idx, row in comparison_df.iterrows():
            report_lines.append(f"{idx+1:2d}. {row['Model']:15s} - {row['Mean Accuracy']}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-"*15)
        report_lines.append(f"1. Best single model: {best_model}")
        
        # Check for ensemble potential
        top3_models = comparison_df.head(3)['Model'].tolist()
        report_lines.append(f"2. Consider ensemble of top models: {', '.join(top3_models)}")
        
        # Stability analysis
        most_stable = comparison_df.loc[comparison_df['Std Dev'].astype(float).idxmin(), 'Model']
        report_lines.append(f"3. Most stable model: {most_stable}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Analysis report saved to {save_path}")
        
        # Print summary to console
        print("\nSUMMARY:")
        print(f"Best model: {best_model} ({best_acc:.4f})")
        print(f"Most stable: {most_stable}")
        print(f"Full report: {save_path}")
        
        return save_path

def main():
    """Main analysis pipeline."""
    
    print("Cross-Validation Results Analyzer")
    print("="*50)
    
    # Configuration
    data_file = "data_quantile_Target_QC_aggcat"
    k_folds = 5
    
    # Initialize analyzer
    analyzer = CVResultsAnalyzer()
    
    # Load results
    analyzer.load_cv_results(data_file, k_folds)
    
    if not analyzer.results_data:
        print("No CV results found. Make sure cross-validation has been completed.")
        return
    
    print(f"\nLoaded results for {len(analyzer.results_data)} models")
    
    # Generate analysis
    print("\nGenerating analysis...")
    
    # Summary statistics
    summary_stats = analyzer.create_summary_statistics()
    comparison_df = analyzer.create_detailed_comparison()
    
    print("\nModel Performance Summary:")
    print(comparison_df.to_string(index=False))
    
    # Generate plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating visualizations...")
    
    # Accuracy distribution
    acc_dist_path = f"cv_accuracy_distribution_{timestamp}.png"
    analyzer.plot_accuracy_distribution(acc_dist_path)
    
    # Learning curves
    learning_curves_path = f"cv_learning_curves_{timestamp}.png"
    analyzer.plot_learning_curves(learning_curves_path)
    
    # Statistical tests
    print("\nPerforming statistical significance tests...")
    statistical_tests = analyzer.perform_statistical_tests()
    
    if statistical_tests:
        print("Significant differences found:")
        for comparison, result in statistical_tests.items():
            if result['significant']:
                print(f"  {comparison}: p={result['p_value']:.4f}")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report_path = analyzer.generate_report()
    
    print(f"\n✅ Analysis complete!")
    print(f"Report: {report_path}")
    print(f"Plots: {acc_dist_path}, {learning_curves_path}")

if __name__ == "__main__":
    main()