# DLSS-WAQP: Wikipedia Article Quality Prediction with Graph Neural Networks

**Deep Learning for Social Scientists - Final Project**

A comprehensive GNN-based approach to predict Wikipedia article quality using network topology and content features.

[Processed Data for Exploratory Analysis and GNN Training](https://drive.google.com/drive/folders/1QWCKvZHmtQ5PyUHGaGJVG4RbtXtJJkmh?usp=sharing)

## 🚀 Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run optimized training (if optimization results exist)
cd 1_CodePipeline/2_GNN_Training
python 2.2_runner_final.py
```

## 📋 Project Overview

This project implements a state-of-the-art pipeline for Wikipedia article quality prediction using:
- **Advanced Graph Neural Networks** (ImprovedGNN, ResidualGCN, GraphSAGE, GAT, MLP Baseline)
- **Automated Hyperparameter Optimization** with Optuna integration
- **Sophisticated Class Imbalance Handling** with smart oversampling strategies and focal loss variants
- **Memory-Efficient Training** with GraphSAINT sampling for large-scale graphs
- **Enhanced Training Pipeline** with automatic optimization result integration
- **Device-Optimized Implementation** with full MPS/CUDA/CPU support

## 🔥 **Recent Major Updates**

### ✅ **Fixed Critical Issues**
- **Device Placement Bug Fixes**: Resolved tensor device mismatch errors causing -inf scores
- **GAT Model Architecture Fix**: Corrected layer dimensions and forward pass logic
- **GAT Learning Rate Optimization**: Added model-specific learning rate constraints (1e-4 to 1e-2)

### ✅ **Enhanced Training Pipeline** 
- **Intelligent Parameter Collection**: Automatic optimization result integration
- **Smart Fallback System**: Sensible defaults when optimization results unavailable
- **Enhanced Runner Scripts**: `2.2_runner.py` now automatically uses optimized hyperparameters

### ✅ **Optimization Improvements**
- **Cross-Validation Fixes**: Proper device handling in memory-efficient sampling
- **Loss Function Enhancements**: Improved smart oversampling with device consistency
- **Memory Management**: Better MPS device handling for Apple Silicon

## 🏗️ Project Structure

```
DLSS-WAQP/
├── 0_TasksDocumentation/           # 📋 Project documentation & task management
│   ├── ProjectRequirements_README.md
│   ├── Project_presentation.odp
│   ├── Steps_Tasks_Resources.md
│   ├── benchmarks.pdf
│   ├── final_feature.csv
│   └── title_pageview_history.ipynb
├── 1_WikiDataNet/                  # 🌐 Data collection & network construction
│   ├── 1.1_WikiData.py            # Wikipedia data collection & preprocessing
│   ├── 1.2_WikiNet.py             # Network construction & feature engineering
│   ├── 1.3_PrepData.py            # Additional data preparation utilities
│   ├── ArticlesDescriptives.ipynb # 📊 Article analysis & visualization
│   ├── NetworkDescriptives.ipynb  # 🕸️  Network topology analysis
│   ├── data_analysis_exploration.ipynb  # 🔍 Exploratory data analysis
│   └── data/                      # 💾 Raw and processed Wikipedia data
│       ├── cleaned_articles_final.csv
│       ├── cleaned_edges_final.csv
│       ├── df_wiki_data.parquet
│       ├── scaled_data_quantile_Target_QC_aggcat.parquet
│       ├── scaled_data_robust_Target_QC_aggcat.parquet
│       ├── wikidata_ready4net.parquet
│       └── ... (multiple scaling variants)
├── 1_CodePipeline/2_GNN_Training/  # 🤖 Advanced GNN training pipeline
│   ├── 2.1_runner_optimization.py # 🎯 Hyperparameter optimization runner
│   ├── 2.2_runner_final.py        # 🚀 Enhanced training runner (main script)
│   ├── 2.3_Eval.py                # 📈 Model evaluation & analysis
│   ├── Enhanced_Training.py       # 💪 Core enhanced training implementation
│   ├── Hyperparameter_Optimization.py  # ⚙️ Optuna-based optimization
│   ├── ores_benchmark.ipynb       # 🏆 Benchmark comparison with ORES
│   ├── utils/                     # 🛠️ Core utility modules
│   │   ├── GNN_model.py           # 🧠 All GNN model architectures
│   │   ├── loss_functions.py      # ⚖️ Advanced loss functions & oversampling
│   │   ├── cross_validation.py    # 🔄 Graph-aware cross-validation
│   │   ├── feature_engineering.py # 🔧 Graph feature computation
│   │   ├── model_loader.py        # 📤 Model saving/loading utilities
│   │   ├── evaluate_gnn_model.py  # 📊 Model evaluation metrics
│   │   ├── train_GNN_model.py     # 🏋️ Core training functions
│   │   └── ... (additional utilities)
│   ├── data/                      # 💾 Processed graph data (.pt files)
│   │   ├── data_quantile_Target_QC_aggcat.pt
│   │   ├── data_robust_Target_QC_aggcat.pt
│   │   ├── data_standard_Target_QC_aggcat.pt
│   │   └── ... (multiple scaling variants)
│   ├── optimization_results/      # 🎯 Hyperparameter optimization results
│   │   ├── best_*_model.pt        # Best model checkpoints
│   │   ├── *_optimization.json    # Detailed optimization results
│   │   └── ... (results for each model type)
│   ├── optimization_logs/         # 📋 Detailed optimization logs
│   │   ├── gat_optimization_*.log
│   │   ├── improved_gnn_optimization_*.log
│   │   └── ... (logs for each model type)
│   ├── optimization_progress/     # 📊 Optuna study progress tracking
│   │   ├── gat_progress.json
│   │   ├── improved_gnn_progress.json
│   │   └── ... (progress for each model)
│   ├── check/                     # 💾 Enhanced model checkpoints
│   │   ├── enhanced_*_with_config.pt  # Models with full configuration
│   │   └── ... (checkpoints for all model types)
│   ├── lossVisual/               # 📈 Training loss visualizations
│   │   ├── enhanced_gat_*.png
│   │   ├── enhanced_improved_gnn_*.png
│   │   └── ... (loss curves for all models)
│   └── requirements.txt          # 📦 GNN-specific dependencies
├── 2_FinalReport/                # 📄 Final research report
│   ├── Final_Report_DLSS.pdf     # 📑 Complete research report
│   ├── Final_Report_DLSS.qmd     # 📝 Quarto source document
│   ├── references.bib            # 📚 Bibliography
│   ├── Images/                   # 🖼️ Report figures & visualizations
│   └── Tables/                   # 📊 Statistical tables & results
├── README.md                     # 📖 Main project documentation
└── requirements.txt              # 📦 Complete project dependencies
```

## ⚡ Complete Project Execution

### Option A: Quick Training (Recommended)
Use pre-optimized hyperparameters for fast training:

```bash
cd 1_CodePipeline/2_GNN_Training
python 2.2_runner_final.py
```

**This will:**
- **Intelligently detect and use** existing optimization results
- **Train all models** with optimized hyperparameters: ImprovedGNN, ResidualGCN, ResidualSAGE, GAT, MLP
- **Apply device-specific optimizations** (MPS/CUDA/CPU)
- **Use GraphSAINT sampling** for memory efficiency on large graphs
- **Provide detailed progress reporting** with parameter summaries
- **Save enhanced models** to `check/` directory with configuration

### Option B: Full Pipeline
Complete optimization and training pipeline:

```bash
cd 1_CodePipeline/2_GNN_Training

# Step 1: Hyperparameter optimization for all models
python 2.1_runner_optimization.py --n_trials 50 --epochs_per_trial 20

# Step 2: Training with optimized parameters
python 2.2_runner_final.py
```

### Option C: Individual Model Training
Train specific models:

```bash
# ImprovedGNN
python Enhanced_Training.py --data_file data_quantile_Target_QC_aggcat --model_type improved_gnn --epochs 100 --use_graphsaint

# ResidualGCN
python Enhanced_Training.py --data_file data_quantile_Target_QC_aggcat --model_type residual_gcn --epochs 100 --use_graphsaint

# GAT (smaller batch size for memory)
python Enhanced_Training.py --data_file data_quantile_Target_QC_aggcat --model_type gat --epochs 100 --use_graphsaint --batch_size 2048

# MLP Baseline
python Enhanced_Training.py --data_file data_quantile_Target_QC_aggcat --model_type mlp --epochs 100
```

### Option D: Individual Hyperparameter Optimization
Optimize specific models:

```bash
python Hyperparameter_Optimization.py --data_file data_quantile_Target_QC_aggcat --model_type improved_gnn --n_trials 30
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
# Run evaluation script
python 2.3_Eval.py

# Or detailed ORES benchmark comparison
jupyter notebook ores_benchmark.ipynb
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
- `improved_gnn`: Enhanced GNN with residual connections, LayerNorm, and GELU activation
- `residual_gcn`: GCN with residual connections and improved architecture
- `residual_sage`: GraphSAGE with residual connections and enhanced aggregation
- `gat`: Graph Attention Network with **fixed architecture** and **optimized learning rates**
- `mlp`: MLP baseline (no graph structure) for comparison

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
# Install PyTorch first, then extensions:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-sparse torch-scatter torch-cluster

# Verify installation:
python -c "import torch, torch_geometric; print('✅ Installation successful')"
```

### Memory Issues
```bash
# Clear GPU/MPS cache
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache() if torch.backends.mps.is_available() else None"

# For MPS issues on Apple Silicon
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Reduce batch size for large graphs
python Enhanced_Training.py --batch_size 1024 --memory_efficient
```

### Common Error Solutions
```bash
# Fix device mismatch errors (now resolved in code)
# - All tensor operations use consistent device placement
# - GAT model architecture issues have been fixed

# For optimization failures (-inf scores):
# - Updated GAT learning rate constraints
# - Fixed smart oversampling device consistency
# - Enhanced cross-validation memory handling
```

## 📦 Graph-Tool Installation Guide

**graph-tool** is an optional advanced graph analysis library that provides high-performance graph algorithms. While not required for basic GNN training, it offers additional capabilities for complex graph operations.

### 🍎 macOS Installation (Recommended Approach)

#### Method 1: Using Homebrew (Easiest)
```bash
# Update Homebrew
brew update

# Install graph-tool
brew install graph-tool

# Verify installation
python3 -c "import graph_tool; print('✅ graph-tool successfully installed')"
```

#### Method 2: Using Conda (Alternative)
```bash
# Create conda environment with Python 3.10 (recommended)
conda create -n graph-env python=3.10
conda activate graph-env

# Install graph-tool from conda-forge
conda install -c conda-forge graph-tool

# Test installation
python -c "import graph_tool; print('✅ graph-tool working')"
```

### 🐧 Linux Installation

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install dependencies
sudo apt install libboost-all-dev libcgal-dev libsparsehash-dev

# Add graph-tool repository
echo "deb https://downloads.skewed.de/apt jammy main" | sudo tee -a /etc/apt/sources.list
wget -qO - https://keys.openpgp.org/vks/v1/by-fingerprint/612DEFB798507F25 | sudo apt-key add -

# Update and install
sudo apt update
sudo apt install python3-graph-tool

# Verify
python3 -c "import graph_tool; print('✅ Installation successful')"
```

#### CentOS/RHEL/Fedora
```bash
# Install EPEL repository (CentOS/RHEL)
sudo yum install epel-release

# Install dependencies
sudo yum install boost-devel CGAL-devel sparsehash-devel

# Use conda for easiest installation
conda install -c conda-forge graph-tool
```

### 🪟 Windows Installation

#### Using Conda (Recommended)
```bash
# Install conda/miniconda first, then:
conda create -n graph-env python=3.10
conda activate graph-env
conda install -c conda-forge graph-tool
```

#### Using WSL (Alternative)
```bash
# Install Windows Subsystem for Linux, then follow Linux instructions
wsl --install
# Follow Ubuntu installation steps above
```

### 🔧 Terminal Configuration & Environment Setup

#### Add to Your Shell Profile (.bashrc, .zshrc, etc.)
```bash
# For proper graph-tool integration
export PYTHONPATH="/usr/lib/python3/dist-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# For conda users
export CONDA_FORGE_GRAPH_TOOL=1

# Memory optimization for large graphs
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

#### Apply Changes
```bash
# Reload your shell configuration
source ~/.bashrc  # or ~/.zshrc
```

### 🛠️ Troubleshooting Common Issues

#### Issue: Import Error on macOS
```bash
# Solution: Install Python via Homebrew
brew install python@3.10
brew install graph-tool

# Update PATH in ~/.zshrc or ~/.bashrc
export PATH="/opt/homebrew/bin:$PATH"
export PATH="/opt/homebrew/opt/python@3.10/bin:$PATH"
```

#### Issue: GTK/Cairo Dependencies Missing
```bash
# macOS
brew install gtk+3 cairo pycairo pygobject3

# Ubuntu/Debian
sudo apt install libgtk-3-dev libcairo2-dev python3-cairo-dev python3-gi-dev
```

#### Issue: Boost Library Conflicts
```bash
# Remove conflicting installations
brew uninstall boost  # macOS
sudo apt remove libboost-all-dev  # Ubuntu

# Reinstall with graph-tool
brew install graph-tool  # Will install compatible boost version
```

#### Issue: Python Version Incompatibility
```bash
# Use Python 3.8-3.11 (3.10 recommended)
python --version

# If version is incompatible, install correct version:
# macOS
brew install python@3.10

# Ubuntu
sudo apt install python3.10-dev
```

### 🧪 Verify Complete Installation

Create a test script to verify everything works:

```bash
# Create test file
cat > test_graph_tool.py << 'EOF'
#!/usr/bin/env python3

try:
    import graph_tool as gt
    import graph_tool.topology
    import graph_tool.centrality
    
    print("✅ graph-tool imported successfully")
    print(f"📦 Version: {gt.__version__}")
    
    # Create a simple test graph
    g = gt.Graph(directed=False)
    v1 = g.add_vertex()
    v2 = g.add_vertex()
    e = g.add_edge(v1, v2)
    
    print(f"🔗 Test graph created: {g.num_vertices()} vertices, {g.num_edges()} edges")
    print("🎉 graph-tool installation fully functional!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("💡 Try reinstalling with: brew install graph-tool")
    
except Exception as e:
    print(f"⚠️  Partial installation - some features may not work: {e}")
EOF

# Run test
python3 test_graph_tool.py
```

### 🚀 Integration with DLSS-WAQP Project

Once installed, you can leverage graph-tool in your analyses:

```python
# Example usage in network analysis
import graph_tool as gt
import graph_tool.topology
import networkx as nx

# Convert NetworkX graph to graph-tool (much faster for large graphs)
def networkx_to_graph_tool(G_nx):
    g = gt.Graph(directed=G_nx.is_directed())
    
    # Add vertices
    vertex_map = {}
    for node in G_nx.nodes():
        v = g.add_vertex()
        vertex_map[node] = v
    
    # Add edges
    for edge in G_nx.edges():
        g.add_edge(vertex_map[edge[0]], vertex_map[edge[1]])
    
    return g, vertex_map

# Use in your analysis pipeline
# G_nx = your NetworkX graph from WikiNet
# g_gt, vertex_map = networkx_to_graph_tool(G_nx)
# 
# # Fast centrality computation
# betweenness = gt.centrality.betweenness(g_gt)
# closeness = gt.centrality.closeness(g_gt)
```

### 📋 Final Verification Checklist

- [ ] graph-tool imports without errors
- [ ] Version shows correctly (`gt.__version__`)
- [ ] Can create and manipulate graphs
- [ ] All dependencies (GTK, Cairo) working
- [ ] Terminal environment properly configured
- [ ] Test script runs successfully

**Need Help?** Check the [official documentation](https://graph-tool.skewed.de/) or open an issue in the project repository.

### Monitor Training
```bash
# View training progress
tail -f optimization_logs/*.log

# Check GPU usage
python -c "import torch; print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU/MPS\"}')"
```

## 📚 Advanced Research Features

### **🧠 Intelligent Training System**
- **Automatic Optimization Integration**: Seamlessly uses pre-computed hyperparameters
- **Device-Aware Training**: Optimized for MPS (Apple Silicon), CUDA, and CPU
- **Memory-Efficient Architecture**: GraphSAINT sampling for large-scale graphs

### **⚖️ Sophisticated Class Imbalance Handling**
- **Smart Oversampling Strategies**: Balanced, boosted, and adaptive approaches
- **Advanced Loss Functions**: Class-balanced focal loss, standard focal loss, weighted CE
- **Minority Class Optimization**: Specialized metrics for rare class performance

### **🔬 Robust Evaluation Framework**  
- **Graph-Aware Cross-Validation**: Preserves network connectivity in data splits
- **Multiple Performance Metrics**: Accuracy, F1-score, precision, recall for each class
- **Statistical Significance Testing**: Cross-validation with confidence intervals

### **🚀 Performance Optimizations**
- **Model-Specific Hyperparameter Ranges**: GAT uses lower learning rates (1e-4 to 1e-2) for stability
- **Enhanced Device Management**: Fixed tensor device placement issues across all operations
- **Gradient Handling**: Enhanced gradient clipping and normalization
- **Batch Processing**: Intelligent batch size selection based on model type and available memory

### **🛠️ Technical Improvements**
- **Fixed GAT Architecture**: Corrected layer dimensions and forward pass for all layer configurations
- **Device Consistency**: All `torch.where()` and `torch.randperm()` operations use proper device placement
- **Smart Parameter Loading**: Enhanced runner automatically detects and applies optimization results
- **Cross-Platform Compatibility**: Robust handling of MPS (Apple Silicon), CUDA, and CPU devices

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