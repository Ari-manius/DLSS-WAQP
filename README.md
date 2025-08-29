# DLSS-WAQP: Wikipedia Article Quality Prediction

**Deep Learning for Social Scientists - Final Project**

Graph Neural Network-based approach to predict Wikipedia article quality using network topology and content features.

[Processed Data](https://drive.google.com/drive/folders/1QWCKvZHmtQ5PyUHGaGJVG4RbtXtJJkmh?usp=sharing)

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd 1_CodePipeline/2_GNN_Training
python 2.2_runner_final.py
```

## Overview

Implementation features:
- Graph Neural Networks (ImprovedGNN, ResidualGCN, GraphSAGE, GAT, MLP)
- Hyperparameter optimization with Optuna
- Class imbalance handling with oversampling and focal loss
- Memory-efficient training with GraphSAINT sampling
- Cross-platform support (MPS/CUDA/CPU)

## Project Structure

```
DLSS-WAQP/
├── 1_WikiDataNet/                  # Data collection & network construction
│   ├── 1.1_WikiData.py            # Wikipedia data collection
│   ├── 1.2_WikiNet.py             # Network construction & features
│   └── data/                      # Raw and processed data
├── 1_CodePipeline/2_GNN_Training/  # GNN training pipeline
│   ├── 2.1_runner_optimization.py # Hyperparameter optimization
│   ├── 2.2_runner_final.py        # Main training runner
│   ├── 2.3_Eval.py                # Model evaluation
│   ├── Enhanced_Training.py       # Core training implementation
│   ├── utils/                     # Utility modules
│   └── data/                      # Processed graph data
├── 2_FinalReport/                  # Research report
└── requirements.txt               # Dependencies
```

## Usage

### Quick Training
```bash
cd 1_CodePipeline/2_GNN_Training
python 2.2_runner_final.py
```

### Full Pipeline
```bash
# Hyperparameter optimization
python 2.1_runner_optimization.py --n_trials 50

# Training with optimized parameters
python 2.2_runner_final.py
```

### Data Collection
```bash
cd 1_WikiDataNet
python 1.1_WikiData.py
python 1.2_WikiNet.py
```

## Configuration

Key parameters:
- `--scaling_method`: Data preprocessing method
- `--model_type`: GNN architecture (improved_gnn, residual_gcn, gat, mlp)
- `--epochs`: Training epochs
- `--use_graphsaint`: Enable memory-efficient sampling

## Hardware Requirements

- RAM: 16GB+ recommended
- Storage: 5GB+ free space  
- GPU: Optional (CUDA/MPS support)

## Citation

```bibtex
@misc{dlss-waqp-2024,
  title={Wikipedia Article Quality Prediction with Graph Neural Networks},
  author={[Your Name]},
  year={2024},
  note={Deep Learning for Social Scientists - Final Project}
}
```

---

**Keywords**: Graph Neural Networks, Wikipedia, Quality Prediction, PyTorch Geometric, Hyperparameter Optimization