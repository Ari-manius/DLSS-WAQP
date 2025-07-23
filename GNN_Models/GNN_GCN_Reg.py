import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from utils.create_split_masks import create_split_masks_regression
from utils.GNN_model import GNNRegression
from utils.earlyStopping import EarlyStopping
from utils.train_GNN_model import train_GNN_model
from utils.initialize_weights import initialize_weights
import argparse
import os

DATA_DIR = "data"

parser = argparse.ArgumentParser(description="Train a GNN regression model.")
parser.add_argument('--data_file', type=str, required=True, help='Base name of the .pt file (without extension)')
args = parser.parse_args()

# Construct full file path by appending ".pt"
data_path = os.path.join(DATA_DIR, args.data_file + ".pt")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

# Load data
data = torch.load(data_path, weights_only=False)
train_mask, val_mask, test_mask = create_split_masks_regression(data)
data.y = data.y.view(-1, 1).float()

device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
input_dim = data.num_node_features
hidden_dim = 64
output_dim = 1

model = GNNRegression(input_dim, hidden_dim, output_dim).to(device)

print(model)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

initialize_weights(model)
model = torch.compile(model)

optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0001)

warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=10)
cosine = CosineAnnealingLR(optimizer, T_max=300 - 10, eta_min=1e-5)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])

criterion = nn.L1Loss()
#criterion = nn.MSELoss()
#criterion = nn.SmoothL1Loss()

early_stopper = EarlyStopping(patience=30, min_delta=0.0001, path=f'check/GNN_GCN_checkpoint_{args.data_file}.pt')

train_losses, val_losses = train_GNN_model(
    300,
    model,
    optimizer,
    criterion,
    data,
    early_stopper,
    scheduler=scheduler,
    checkpoint_path=f'check/GNN_GCN_checkpoint_{args.data_file}.pt',
    device=device
)

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"lossVisual/TrainVal_Loss_GNN_GCN_{args.data_file}.png")
plt.close()