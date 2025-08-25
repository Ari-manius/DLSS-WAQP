#!/usr/bin/env python3
"""
Ensemble Model Predictor
Creates ensemble predictions from trained cross-validation models.
"""

import os
import json
import pickle
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class EnsemblePredictor:
    """Ensemble predictor that combines multiple model predictions."""
    
    def __init__(self, results_dir="cv_results"):
        self.results_dir = results_dir
        self.model_predictions = {}
        self.true_labels = None
        self.ensemble_predictions = None
        
    def load_fold_predictions(self, model_types, data_file, k_folds):
        """Load predictions from all models and folds."""
        self.model_predictions = defaultdict(list)
        
        for model_type in model_types:
            print(f"Loading predictions for {model_type}...")
            
            for fold in range(k_folds):
                pred_file = os.path.join(
                    self.results_dir, 
                    f"{model_type}_{data_file}_fold_{fold}_predictions.pkl"
                )
                
                if os.path.exists(pred_file):
                    with open(pred_file, 'rb') as f:
                        pred_data = pickle.load(f)
                        
                    if pred_data['predictions'] is not None:
                        self.model_predictions[model_type].append(pred_data['predictions'])
                        
                        # Store true labels (should be same across all folds/models)
                        if self.true_labels is None and pred_data['true_labels'] is not None:
                            self.true_labels = pred_data['true_labels']
                else:
                    print(f"Warning: Missing predictions file {pred_file}")
        
        print(f"Loaded predictions from {len(self.model_predictions)} models")
        for model_type, predictions in self.model_predictions.items():
            print(f"  {model_type}: {len(predictions)} folds")
    
    def create_model_averages(self):
        """Average predictions across folds for each model."""
        model_avg_predictions = {}
        
        for model_type, fold_predictions in self.model_predictions.items():
            if fold_predictions:
                # Convert predictions to probabilities if they're class indices
                processed_predictions = []
                
                for pred in fold_predictions:
                    if len(pred.shape) == 1:  # Class indices
                        # Convert to one-hot encoding
                        num_classes = len(np.unique(self.true_labels))
                        one_hot = np.zeros((len(pred), num_classes))
                        one_hot[np.arange(len(pred)), pred] = 1
                        processed_predictions.append(one_hot)
                    else:  # Already probabilities
                        processed_predictions.append(pred)
                
                # Average across folds
                model_avg_predictions[model_type] = np.mean(processed_predictions, axis=0)
        
        return model_avg_predictions
    
    def create_ensemble_predictions(self, method='average', weights=None):
        """Create ensemble predictions using specified method."""
        model_avg_predictions = self.create_model_averages()
        
        if not model_avg_predictions:
            print("No model predictions available for ensemble")
            return None
        
        if method == 'average':
            # Simple average
            if weights is None:
                weights = {model: 1.0 for model in model_avg_predictions.keys()}
            
            # Normalize weights
            total_weight = sum(weights.values())
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            # Weighted average
            ensemble_probs = None
            for model_type, probs in model_avg_predictions.items():
                weight = normalized_weights.get(model_type, 0)
                if ensemble_probs is None:
                    ensemble_probs = weight * probs
                else:
                    ensemble_probs += weight * probs
                    
            self.ensemble_predictions = np.argmax(ensemble_probs, axis=1)
            
        elif method == 'voting':
            # Majority voting
            votes = []
            for probs in model_avg_predictions.values():
                votes.append(np.argmax(probs, axis=1))
            
            votes = np.array(votes)
            self.ensemble_predictions = np.array([
                np.bincount(votes[:, i]).argmax() 
                for i in range(votes.shape[1])
            ])
        
        elif method == 'weighted_voting':
            # Weighted voting based on individual model performance
            if weights is None:
                weights = {model: 1.0 for model in model_avg_predictions.keys()}
            
            ensemble_votes = np.zeros((len(self.true_labels), len(np.unique(self.true_labels))))
            
            for model_type, probs in model_avg_predictions.items():
                weight = weights.get(model_type, 1.0)
                predictions = np.argmax(probs, axis=1)
                
                for i, pred in enumerate(predictions):
                    ensemble_votes[i, pred] += weight
            
            self.ensemble_predictions = np.argmax(ensemble_votes, axis=1)
        
        return self.ensemble_predictions
    
    def evaluate_ensemble(self):
        """Evaluate ensemble performance."""
        if self.ensemble_predictions is None or self.true_labels is None:
            print("No ensemble predictions or true labels available")
            return None
        
        accuracy = accuracy_score(self.true_labels, self.ensemble_predictions)
        report = classification_report(self.true_labels, self.ensemble_predictions)
        conf_matrix = confusion_matrix(self.true_labels, self.ensemble_predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return results
    
    def evaluate_individual_models(self):
        """Evaluate individual model performance."""
        model_avg_predictions = self.create_model_averages()
        individual_results = {}
        
        for model_type, probs in model_avg_predictions.items():
            predictions = np.argmax(probs, axis=1)
            accuracy = accuracy_score(self.true_labels, predictions)
            
            individual_results[model_type] = {
                'accuracy': accuracy,
                'predictions': predictions.tolist()
            }
            
            print(f"{model_type:15}: {accuracy:.4f}")
        
        return individual_results
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix for ensemble predictions."""
        if self.ensemble_predictions is None or self.true_labels is None:
            print("No predictions available for plotting")
            return
        
        conf_matrix = confusion_matrix(self.true_labels, self.ensemble_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=np.unique(self.true_labels),
                   yticklabels=np.unique(self.true_labels))
        plt.title('Ensemble Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, save_path=None):
        """Plot comparison of individual model accuracies."""
        individual_results = self.evaluate_individual_models()
        ensemble_accuracy = accuracy_score(self.true_labels, self.ensemble_predictions)
        
        models = list(individual_results.keys()) + ['Ensemble']
        accuracies = [individual_results[model]['accuracy'] for model in individual_results.keys()]
        accuracies.append(ensemble_accuracy)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['skyblue'] * len(individual_results) + ['coral'])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_ensemble_results(self, data_file, method='average'):
        """Save ensemble results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Evaluate models
        ensemble_results = self.evaluate_ensemble()
        individual_results = self.evaluate_individual_models()
        
        results = {
            'timestamp': timestamp,
            'data_file': data_file,
            'ensemble_method': method,
            'ensemble_results': ensemble_results,
            'individual_results': individual_results,
            'ensemble_predictions': self.ensemble_predictions.tolist(),
            'true_labels': self.true_labels.tolist()
        }
        
        results_file = os.path.join(
            self.results_dir,
            f"ensemble_results_{data_file}_{method}_{timestamp}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Ensemble results saved to {results_file}")
        return results_file

def main():
    """Main ensemble prediction pipeline."""
    
    print("Ensemble Model Predictor")
    print("=" * 50)
    
    # Configuration
    data_file = "data_quantile_Target_QC_aggcat"
    model_types = ['improved_gnn', 'residual_gcn', 'residual_sage', 'gat', 'mlp']
    k_folds = 5
    
    # Initialize ensemble predictor
    ensemble = EnsemblePredictor()
    
    # Load predictions
    print("Loading cross-validation predictions...")
    ensemble.load_fold_predictions(model_types, data_file, k_folds)
    
    if not ensemble.model_predictions:
        print("No predictions found. Make sure cross-validation training has been completed.")
        return
    
    # Create ensemble predictions using different methods
    methods = ['average', 'voting', 'weighted_voting']
    best_method = None
    best_accuracy = 0
    
    print("\nTesting ensemble methods:")
    print("-" * 30)
    
    for method in methods:
        print(f"\nMethod: {method}")
        predictions = ensemble.create_ensemble_predictions(method=method)
        
        if predictions is not None:
            results = ensemble.evaluate_ensemble()
            accuracy = results['accuracy']
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
    
    # Use best method for final ensemble
    print(f"\nBest ensemble method: {best_method} (Accuracy: {best_accuracy:.4f})")
    ensemble.create_ensemble_predictions(method=best_method)
    
    # Generate detailed results
    print("\n" + "="*50)
    print("FINAL ENSEMBLE RESULTS")
    print("="*50)
    
    print("\nIndividual Model Performance:")
    print("-" * 30)
    individual_results = ensemble.evaluate_individual_models()
    
    print("\nEnsemble Performance:")
    print("-" * 20)
    ensemble_results = ensemble.evaluate_ensemble()
    
    # Save results
    results_file = ensemble.save_ensemble_results(data_file, best_method)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Confusion matrix
    conf_matrix_path = f"ensemble_confusion_matrix_{timestamp}.png"
    ensemble.plot_confusion_matrix(conf_matrix_path)
    
    # Model comparison
    comparison_path = f"model_comparison_{timestamp}.png"
    ensemble.plot_model_comparison(comparison_path)
    
    print(f"\n✅ Ensemble prediction complete!")
    print(f"Results saved: {results_file}")
    print(f"Plots saved: {conf_matrix_path}, {comparison_path}")

if __name__ == "__main__":
    main()