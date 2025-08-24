from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm, InnerProductDecoder, GATConv  # Graph Convolutional Network layer implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

## Old Models
# class GNN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         """
#         Initialize a 2-layer Graph Convolutional Network.

#         Args:
#             input_dim: Number of input features per node
#             hidden_dim: Size of hidden layer (number of features after first convolution)
#             output_dim: Number of output classes (final prediction dimension)
#         """
#         super(GNN, self).__init__()  # Initialize the parent class (torch.nn.Module)

#         # First graph convolutional layer
#         # Transforms node features from input_dim to hidden_dim dimensions
#         self.conv1 = GCNConv(input_dim, hidden_dim)

#         # Second graph convolutional layer
#         # Transforms node features from hidden_dim to output_dim dimensions
#         self.conv2 = GCNConv(hidden_dim, output_dim)


#     def forward(self, data):
#         """
#         Forward pass through the network.

#         Args:
#             data: PyTorch Geometric Data object containing:
#                 - x: Node feature matrix with shape [num_nodes, input_dim]
#                 - edge_index: Graph connectivity in COO format with shape [2, num_edges]
#                   where each column [src, dst] represents an edge

#         Returns:
#             Log probabilities for each class per node
#         """
#         # Extract node features and the graph structure
#         x, edge_index = data.x, data.edge_index

#         # First convolution layer:
#         # For each node, aggregate features from its neighbors following the GCN formula
#         x = self.conv1(x, edge_index)

#         # Apply ReLU activation to introduce non-linearity
#         x = F.relu(x)

#         # Apply dropout for regularization (only active during training)
#         # Randomly zeroes some elements to prevent overfitting
#         x = F.dropout(x, training=self.training)

#         # Second convolution layer:
#         # Further transform node features using neighbor information
#         x = self.conv2(x, edge_index)

#         return x
    
# class GNNRegression(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim=1):
#         super(GNNRegression, self).__init__()

#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
#         self.conv3 = GCNConv(hidden_dim // 2, output_dim)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv3(x, edge_index)

#         # For regression, output raw continuous values
#         return x  # shape: [num_nodes, output_dim]
    
# class GraphSAGE_Classification(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
#         super().__init__()

#         self.conv1 = SAGEConv(input_dim, hidden_dim)
#         self.bn1 = BatchNorm(hidden_dim)

#         self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
#         self.bn2 = BatchNorm(hidden_dim // 2)

#         self.conv3 = SAGEConv(hidden_dim // 2, output_dim)  # output_dim=2 for binary, >2 for multiclass

#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv3(x, edge_index)

#         return x  # raw logits

# class GraphSAGE_Regression(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, dropout=0.5):
#         super().__init__()

#         self.conv1 = SAGEConv(input_dim, hidden_dim)
#         self.bn1 = BatchNorm(hidden_dim)

#         self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
#         self.bn2 = BatchNorm(hidden_dim // 2)

#         self.conv3 = SAGEConv(hidden_dim // 2, 1)  # Output single value for regression

#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv3(x, edge_index)

#         return x  # keep shape [num_nodes, 1] to match target

# class ImprovedGraphSAGEReg(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
#         super().__init__()

#         self.conv1 = SAGEConv(input_dim, hidden_dim)
#         self.bn1 = LayerNorm(hidden_dim)

#         self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
#         self.bn2 = LayerNorm(hidden_dim // 2)

#         self.conv3 = SAGEConv(hidden_dim // 2, 1)

#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.gelu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.gelu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv3(x, edge_index)
#         return x  # keep shape [num_nodes, 1] to match target

##Anomaly Detection (Experiment)
# Only Nodes
class AnomalyGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, data):
        z = F.relu(self.encoder(data.x, data.edge_index))
        x_recon = self.decoder(z)
        return z, x_recon

# Full GAE - Nodes and Edges 
class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.decoder = InnerProductDecoder()  # Reconstructs edges too

    def encode(self, x, edge_index):
        return F.relu(self.encoder(x, edge_index))

    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))  # Edge probabilities


## New Models 
class ResidualGCN(torch.nn.Module):
    """
    GCN with residual connections and improved architecture for classification.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(ResidualGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection layer with improved initialization
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
        
        # Output layer
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Skip connection projections for dimension matching
        self.skip_projs = nn.ModuleList()
        for i in range(num_layers):
            self.skip_projs.append(nn.Identity())
            
        # Note: Weights initialized by external initialize_weights function

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection with better activation
        x = self.input_proj(x)
        x = F.gelu(x)  # GELU is smoother than ReLU
        
        # GCN layers with residual connections
        for i in range(self.num_layers):
            identity = self.skip_projs[i](x)
            
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.gelu(x)  # Consistent activation
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Weighted residual connection for better gradient flow
            alpha = 0.9  # Slightly favor new information
            x = alpha * x + (1 - alpha) * identity
        
        # Output projection
        x = self.output_proj(x)
        return x


class ResidualGraphSAGE(torch.nn.Module):
    """
    GraphSAGE with residual connections and improved architecture.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(ResidualGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # SAGE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
        
        # Output layer
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection with normalization
        x = self.input_proj(x)
        x = F.gelu(x)
        
        # SAGE layers with residual connections
        for i in range(self.num_layers):
            identity = x
            
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Weighted residual connection
            alpha = 0.85  # Slightly different weight than GCN
            x = alpha * x + (1 - alpha) * identity
        
        # Output projection
        x = self.output_proj(x)
        return x


class GraphAttentionNet(torch.nn.Module):
    """
    Graph Attention Network (GAT) implementation for both classification and regression.
    Enhanced with better dropout strategies and attention mechanisms.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, num_layers=2, dropout=0.3):
        super(GraphAttentionNet, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_dropout = max(0.1, dropout - 0.1)  # Separate attention dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        if num_layers == 1:
            # Single layer case: direct input to output mapping
            self.convs.append(GATConv(input_dim, output_dim, heads=1, dropout=self.attention_dropout, concat=False))
            self.norms.append(LayerNorm(output_dim))
        else:
            # Multi-layer case: build layers progressively
            # First layer with separate attention dropout
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=self.attention_dropout, concat=True))
            self.norms.append(LayerNorm(hidden_dim * heads))
            
            # Hidden layers
            for i in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=self.attention_dropout, concat=True))
                self.norms.append(LayerNorm(hidden_dim * heads))
            
            # Final layer (no concatenation, average attention heads)
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=min(2, heads), dropout=self.attention_dropout, concat=False))
            
        # Note: Weights initialized by external initialize_weights function

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.num_layers == 1:
            # Single layer: apply with normalization for consistency
            x = self.convs[0](x, edge_index)
            x = self.norms[0](x)  # Apply normalization for single layer too
            return x
        else:
            # Multi-layer: apply all hidden layers with activation/dropout, then final layer
            for i in range(len(self.convs) - 1):
                x = self.convs[i](x, edge_index)
                x = self.norms[i](x)
                x = F.elu(x)  # ELU works well with attention
                # Use different dropout rates for different layers
                layer_dropout = self.dropout * (0.8 + 0.2 * i / max(1, len(self.convs) - 2))
                x = F.dropout(x, p=layer_dropout, training=self.training)
            
            # Final layer (reduced dropout)
            x = self.convs[-1](x, edge_index)
            x = F.dropout(x, p=self.dropout * 0.5, training=self.training)
            
            return x


class ImprovedGNN(torch.nn.Module):
    """
    Improved GNN combining multiple techniques:
    - Residual connections
    - Layer normalization
    - GELU activation
    - Dropout
    - Multiple aggregation functions
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.4, use_sage=True):
        super(ImprovedGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_sage = use_sage
        
        # Feature preprocessing
        self.input_norm = LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        Conv = SAGEConv if use_sage else GCNConv
        
        for i in range(num_layers):
            self.convs.append(Conv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
            self.skip_connections.append(nn.Identity())
        
        # Output layers
        self.output_norm = LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Note: Weights initialized by external initialize_weights function
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input preprocessing
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph convolution with residual connections
        for i in range(self.num_layers):
            identity = self.skip_connections[i](x)
            
            # Graph convolution
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.gelu(x)
            
            # Adaptive dropout - less in later layers for stability
            layer_dropout = self.dropout * (1.0 - 0.1 * i / max(1, self.num_layers - 1))
            x = F.dropout(x, p=layer_dropout, training=self.training)
            
            # Weighted residual connection for better gradient flow
            alpha = 0.8 + 0.1 * i / max(1, self.num_layers - 1)  # Slightly more residual in later layers
            x = alpha * x + (1 - alpha) * identity
        
        # Output
        x = self.output_norm(x)
        x = self.output_proj(x)
        
        return x


class MLPBaseline(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) baseline model.
    Uses only node features without considering graph structure.
    Serves as a benchmark to evaluate the benefit of graph information.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.4):
        super(MLPBaseline, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input preprocessing
        self.input_norm = LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
        
        # Output layer
        self.output_norm = LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Note: Weights initialized by external initialize_weights function
    
    def forward(self, data):
        # Only use node features, ignore graph structure
        x = data.x
        
        # Input preprocessing
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers with residual connections
        for i in range(len(self.layers)):
            identity = x
            
            x = self.layers[i](x)
            x = self.norms[i](x)
            x = F.gelu(x)
            
            # Adaptive dropout for MLP too
            layer_dropout = self.dropout * (0.9 + 0.1 * i / max(1, len(self.layers) - 1))
            x = F.dropout(x, p=layer_dropout, training=self.training)
            
            # Weighted residual connection
            alpha = 0.85
            x = alpha * x + (1 - alpha) * identity
        
        # Output
        x = self.output_norm(x)
        x = self.output_proj(x)
        
        return x