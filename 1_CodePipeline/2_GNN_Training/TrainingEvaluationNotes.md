Train/Test/Validation-Split
- Hyperparameter Optimization: 70/15/15 (single split)
- Final Training with CV: 80/10/10 (different splits per run)

Class Imbalance Handling
- Loss functions: Focal Loss (α=1.0-3.0, γ=2.0-4.0), Class-Balanced Focal Loss (β=0.999-0.9999), Weighted Cross-Entropy
- Smart oversampling strategies:
  - "balanced": Equal samples per class
  - "boosted": Aggressive minority class boosting (2.0-5.0x factor)
- Implementation: Memory-safe with 20,000 samples max, weighted sampling support
- Effective number calculation for class-balanced losses using beta parameter

Training Configuration
- Early stopping: 25-50 patience, min_delta=1e-4, best model checkpointing
- Optimization: Adam optimizer with OneCycleLR scheduling
  - Max LR: optimized value, PCT start: 0.1 (10% warmup)
  - Anneal strategy: cosine, fallback: ReduceLROnPlateau
- Regularization: Gradient clipping (max_norm=1.0), dropout, model compilation
- Memory management: 
  - GraphSAINT subgraph sampling: batch size 2048-8192
  - Walk length: 2, sampling steps: 5-8 per epoch
  - MPS device fallback handling for Apple Silicon

Hyperparameter Optimization (Optuna)
- Framework: Optuna with TPE (Tree-structured Parzen Estimator) sampler
- Search spaces:
  - Universal: Dropout (0.01-0.7), Weight decay (1e-6 to 1e-2 log scale)
  - Learning rate: 1e-4 to 1e-1 (standard), 1e-4 to 1e-2 (GAT)
  - Hidden dimensions: [32, 64, 128, 256] categorical
  - Layers: 2-5 (GAT: 2-3 for stability)
- Model-specific: GAT heads [2,4,8,16], Loss functions ['focal', 'class_balanced_focal', 'weighted_ce']
- Trials: 16-100 per model (typically 16-50)
- Epochs: 40 per trial (optimization), 80-200 (final training)
- Objective: Combined metric (60% minority F1-score + 40% overall accuracy)
- Direction: Maximize for classification tasks

Cross-Validation Configuration
- Implementation: Graph-aware splitting with connectivity preservation
- Hyperparameter Optimization: 3-fold stratified CV (classification)
- Data splits: 
  - Optimization phase: 70/15/15 (train/val/test)
  - Final training: 80/10/10 (train/val/test)
- Final Training: 3 runs per model with different random seeds
- Random Seeds: Base seed 42, variants (42, 142, 242) for reproducibility
- Memory-efficient subgraph sampling for large graphs
- Stratified sampling ensures class distribution consistency
- Epochs: 40 (optimization trials), 150 (final training runs)

Evaluation Metrics
- Classification: Accuracy, Precision, Recall, F1-score (macro/weighted)
- Cross-validation: Mean ± std across runs with consistent data splits
- Class-specific: Per-class performance, minority class focus
- Evaluation uses same CV splits as training for consistent results


Model-Specific Best Parameters
ResidualGraphSAGE (Best Overall - Score: 0.8108)
- Dropout: 0.298, Weight decay: 0.0073, Learning rate: 0.00128
- Hidden dim: 128, Layers: 2
- Loss: Focal (α=1.00, γ=3.98), Oversample: balanced, Factor: 4

ImprovedGNN (Score: 0.7992)
- Dropout: 0.110, Weight decay: 4.2e-06, Learning rate: 0.0173
- Hidden dim: 64, Layers: 3
- Loss: Class-Balanced Focal (β=0.9993, γ=3.13, boost=4.84)

MLP Baseline - Network Data (Score: 0.8101)
- Dropout: 0.177, Weight decay: 1.2e-04, Learning rate: 0.0078
- Hidden dim: 256, Layers: 2
- Loss: Weighted Cross-Entropy, Oversample: boosted, Factor: 2

MLP Baseline - Non-Network Data (Score: 0.8052)
- Dropout: 0.014, Weight decay: 2.5e-04, Learning rate: 0.0014
- Hidden dim: 128, Layers: 3
- Loss: Class-Balanced Focal (β=0.9999, γ=2.55, boost=4.99)

ResidualGCN (Score: 0.7823)
- Dropout: 0.015, Weight decay: 2.9e-05, Learning rate: 0.0017
- Hidden dim: 256, Layers: 2
- Loss: Focal (α=1.11, γ=3.95), Oversample: boosted, Factor: 4

GAT - Poorest Performance (Score: 0.4051)
- Dropout: 0.638, Weight decay: 7.5e-05, Learning rate: 0.00043
- Hidden dim: 128, Layers: 3, Heads: 2
- Loss: Focal (α=2.83, γ=3.78), Oversample: boosted, Factor: 2



