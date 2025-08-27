import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

def load_data_and_create_splits(data_path, test_size=0.2, val_size=0.1, random_state=42):
    """Load data and create train/val/test splits"""
    df = pd.read_parquet(data_path)
    
    # Separate features and target
    target_col = 'Target_QC_aggcat'
    feature_cols = [col for col in df.columns if col not in [target_col, 'node_id']]
    
    X = df[feature_cols].values
    y = df[target_col].values.astype(int)
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, 
        test_size=val_size/(1-test_size), 
        random_state=random_state, 
        stratify=y_trainval
    )
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': feature_cols
    }

def train_random_forest_cv(data_splits, n_runs=3, random_state_base=42):
    """Train Random Forest with cross-validation"""
    
    # Combine train and validation for full training
    X_full_train = np.vstack([data_splits['X_train'], data_splits['X_val']])
    y_full_train = np.hstack([data_splits['y_train'], data_splits['y_val']])
    
    # Random Forest
    rf_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'bootstrap': True,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }
    
    results = []
    confusion_matrices = []
    
    for run in range(1, n_runs + 1):
        print(f"\nRun {run}/{n_runs}:")
        
        # Create model
        rf = RandomForestClassifier(
            **rf_params,
            random_state=random_state_base + run
        )
        
        # Train on full training set
        start_time = time.time()
        rf.fit(X_full_train, y_full_train)
        train_time = time.time() - start_time
        
        # Evaluate on test set
        y_pred = rf.predict(data_splits['X_test'])
        
        # metrics
        accuracy = accuracy_score(data_splits['y_test'], y_pred)
        report = classification_report(data_splits['y_test'], y_pred, output_dict=True)
        cm = confusion_matrix(data_splits['y_test'], y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        
        # Store results
        run_result = (report, cm)
        results.append(run_result)
        confusion_matrices.append(cm)
        
        # Feature importance for first run
        if run == 1:
            # Gini importance (built-in)
            gini_importance = pd.DataFrame({
                'feature': data_splits['feature_names'],
                'gini_importance': rf.feature_importances_
            }).sort_values('gini_importance', ascending=False)
            
            # Permutation importance
            perm_imp = permutation_importance(rf, data_splits['X_test'], data_splits['y_test'], 
                                            n_repeats=5, random_state=42)
            perm_importance = pd.DataFrame({
                'feature': data_splits['feature_names'],
                'perm_importance': perm_imp.importances_mean
            }).sort_values('perm_importance', ascending=False)
            
            print(f"\n10 Most Important Features (Gini Impurity):")
            for i, (_, row) in enumerate(gini_importance.head(10).iterrows()):
                print(f"  {i+1}. {row['feature']}: {row['gini_importance']:.4f}")
                
            print(f"\n10 Most Important Features (Permutation):")
            for i, (_, row) in enumerate(perm_importance.head(10).iterrows()):
                print(f"  {i+1}. {row['feature']}: {row['perm_importance']:.4f}")
    
    return results, confusion_matrices

def calculate_cv_statistics(results):
    """Calculate cross-validation statistics"""
    accuracies = [r[0]['accuracy'] for r in results]
    precisions = [r[0]['macro avg']['precision'] for r in results]
    recalls = [r[0]['macro avg']['recall'] for r in results]
    f1_scores = [r[0]['macro avg']['f1-score'] for r in results]
    
    cv_stats = {
        'n_runs': len(results),
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
    
    return cv_stats

def save_results(cv_stats, confusion_matrices, output_file='random_forest_results.json'):
    """Save results in JSON"""
    
    # numpy arrays to lists
    results_json = {
        'random_forest': {
            'n_runs': cv_stats['n_runs'],
            'accuracy': {
                'mean': float(cv_stats['accuracy']['mean']),
                'std': float(cv_stats['accuracy']['std']),
                'values': [float(x) for x in cv_stats['accuracy']['values']]
            },
            'precision': {
                'mean': float(cv_stats['precision']['mean']),
                'std': float(cv_stats['precision']['std']),
                'values': [float(x) for x in cv_stats['precision']['values']]
            },
            'recall': {
                'mean': float(cv_stats['recall']['mean']),
                'std': float(cv_stats['recall']['std']),
                'values': [float(x) for x in cv_stats['recall']['values']]
            },
            'f1_score': {
                'mean': float(cv_stats['f1_score']['mean']),
                'std': float(cv_stats['f1_score']['std']),
                'values': [float(x) for x in cv_stats['f1_score']['values']]
            }
        },
        'confusion_matrices': [cm.tolist() for cm in confusion_matrices]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return results_json


if __name__ == "__main__":
    # Load data
    data_path = "/Users/lorenzruckert/Desktop/Studium/Master/Viertes Semester/Deep Learning/Final Project/DLSS-WAQP/1_CodePipeline/1_WikiDataNet/data/scaled_data_quantile_Target_QC_aggcat.parquet"

    data_splits = load_data_and_create_splits(data_path)

    # Train and evaluate Random Forest
    results, confusion_matrices = train_random_forest_cv(data_splits, n_runs=3)
    
    # Calculate statistics
    cv_stats = calculate_cv_statistics(results)

    # Save results
    save_results(cv_stats, confusion_matrices, 'random_forest_results.json')