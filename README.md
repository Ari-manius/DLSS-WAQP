# DLSS-WAQP: Wikipedia Article Quality Prediction with Graph Neural Networks

**Deep Learning for Social Scientists - Final Project**

A comprehensive GNN-based approach to predict Wikipedia article quality using network topology and content features.

## 🚀 Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run optimized training (if optimization results exist)
cd 1_CodePipeline/2_GNN_Training
python 2.2_runner.py
```

## 📋 Project Overview

This project implements a complete pipeline for Wikipedia article quality prediction using:
- **Graph Neural Networks** (GCN, GraphSAGE, GAT, ResidualGNN)
- **Hyperparameter Optimization** with Optuna
- **Class Imbalance Handling** with smart oversampling and focal loss
- **Memory-Efficient Training** with GraphSAINT sampling

## 🏗️ Project Structure

```
DLSS-WAQP/
├── 0_TasksDocumentation/           # Project documentation & tasks
├── 1_WikiDataNet/                  # Data collection & network construction
│   ├── 1.1_WikiData.py            # Wikipedia data collection
│   ├── 1.2_WikiNet.py             # Network construction
│   └── data/                      # Raw and processed data
├── 1_CodePipeline/2_GNN_Training/  # GNN training pipeline
│   ├── 2.1_PrepData.py           # Data preprocessing
│   ├── 2.2_Enhanced_Training.py   # Enhanced model training
│   ├── 2.2_runner.py             # Automated training runner
│   ├── 2.3_Hyperparameter_Optimization.py  # Optuna optimization
│   ├── 2.3_runner_with_optimization.py     # Full pipeline runner
│   ├── utils/                     # Utility modules
│   ├── data/                      # Processed graph data
│   ├── optimization_results/      # Hyperparameter results
│   └── check/                     # Model checkpoints
├── 2_FinalReport/                 # Final report & references
└── requirements.txt               # Dependencies
```

## ⚡ Complete Project Execution

### Option A: Quick Training (Recommended)
Use pre-optimized hyperparameters for fast training:

```bash
cd 1_CodePipeline/2_GNN_Training
python 2.2_runner.py
```

**This will:**
- Automatically use optimization results (if available)
- Train all models: ImprovedGNN, ResidualGCN, ResidualSAGE, GAT, MLP
- Apply GraphSAINT sampling for memory efficiency
- Save models to `check/` directory

### Option B: Full Pipeline
Complete optimization and training pipeline:

```bash
cd 1_CodePipeline/2_GNN_Training

# Step 1: Data preparation (if needed)
python 2.1_PrepData.py

# Step 2: Hyperparameter optimization
python 2.3_runner_with_optimization.py --n_trials 50 --epochs_per_trial 20

# Step 3: Training with optimized parameters
python 2.2_runner.py
```

### Option C: Individual Model Training
Train specific models:

```bash
# ImprovedGNN
python 2.2_Enhanced_Training.py --data_file data_quantile_Target_QC_aggcat --model_type improved_gnn --epochs 100 --use_graphsaint

# ResidualGCN
python 2.2_Enhanced_Training.py --data_file data_quantile_Target_QC_aggcat --model_type residual_gcn --epochs 100 --use_graphsaint

# GAT (smaller batch size for memory)
python 2.2_Enhanced_Training.py --data_file data_quantile_Target_QC_aggcat --model_type gat --epochs 100 --use_graphsaint --batch_size 2048

# MLP Baseline
python 2.2_Enhanced_Training.py --data_file data_quantile_Target_QC_aggcat --model_type mlp --epochs 100
```

### Option D: Individual Hyperparameter Optimization
Optimize specific models:

```bash
python 2.3_Hyperparameter_Optimization.py --data_file data_quantile_Target_QC_aggcat --model_type improved_gnn --n_trials 30
```

## 📊 Data Collection & Processing

### Phase 1: Wikipedia Data Collection
```bash
cd 1_WikiDataNet

# Collect and process Wikipedia data
python 1.1_WikiData.py

# Construct network and compute features  
python 1.2_WikiNet.py

# Optional: Exploratory analysis
jupyter notebook ArticlesDescriptives.ipynb
jupyter notebook NetworkDescriptives.ipynb
```

## 🔍 Evaluation & Analysis

```bash
# Detailed evaluation notebook
jupyter notebook 2.4_Eval.ipynb

# Or run evaluation script
python Eval.py
```

## 📈 Project Outputs

After running the pipeline, you'll find:

- **Trained Models**: `check/enhanced_*_with_config.pt`
- **Optimization Results**: `optimization_results/*.json`
- **Training Logs**: `optimization_logs/*.log`
- **Loss Visualizations**: `lossVisual/*.png`
- **Processed Data**: `data/*.pt`

## ⚙️ Configuration Options

### Key Parameters
- **Data preprocessing**: `--scaling_method` (robust, quantile, standard, minmax)
- **Model architecture**: `--hidden_dim`, `--num_layers`, `--dropout`
- **Training**: `--epochs`, `--lr`, `--batch_size`
- **Optimization**: `--n_trials`, `--n_splits`
- **Memory management**: `--use_graphsaint`, `--memory_efficient`

### Model Types
- `improved_gnn`: Enhanced GNN with residual connections
- `residual_gcn`: GCN with residual connections
- `residual_sage`: GraphSAGE with residual connections  
- `gat`: Graph Attention Network
- `mlp`: MLP baseline (no graph structure)

## 🖥️ Hardware Requirements

- **RAM**: 16GB+ recommended (8GB minimum)
- **Storage**: 5GB+ free space
- **GPU**: Optional but recommended (CUDA/MPS support)
- **CPU**: Multi-core recommended for cross-validation

## ⏱️ Expected Runtime

- **Data Collection**: 30-60 minutes
- **Hyperparameter Optimization**: 2-8 hours (depends on n_trials)
- **Model Training**: 1-3 hours (all models)
- **Total Runtime**: 4-12 hours

**For faster execution:**
- Reduce `n_trials` for optimization
- Use fewer epochs for initial testing
- Enable GPU acceleration

## 🔧 Troubleshooting

### Installation Issues
```bash
# For dependency conflicts, try:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-sparse torch-scatter torch-cluster
```

### Memory Issues
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache() if torch.backends.mps.is_available() else None"

# For MPS issues on Apple Silicon
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Monitor Training
```bash
# View training progress
tail -f optimization_logs/*.log

# Check GPU usage
python -c "import torch; print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU/MPS\"}')"
```

## 📚 Research Features

- **Graph-Aware Cross-Validation**: Preserves network structure in splits
- **Class Imbalance Handling**: Smart oversampling + focal loss
- **Memory-Efficient Training**: GraphSAINT sampling for large graphs
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Multiple GNN Architectures**: Comprehensive model comparison

## 🔬 Reproducibility

For reproducible results:
```bash
export PYTHONHASHSEED=42
python <script> --random_state 42

# Save exact environment
pip freeze > exact_requirements.txt
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{dlss-waqp-2024,
  title={Wikipedia Article Quality Prediction with Graph Neural Networks},
  author={[Your Name]},
  year={2024},
  note={Deep Learning for Social Scientists - Final Project}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.

---

**Keywords**: Graph Neural Networks, Wikipedia, Quality Prediction, PyTorch Geometric, Hyperparameter Optimization, Class Imbalance, Social Science Research