from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm, InnerProductDecoder  # Graph Convolutional Network layer implementation
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize a 2-layer Graph Convolutional Network.

        Args:
            input_dim: Number of input features per node
            hidden_dim: Size of hidden layer (number of features after first convolution)
            output_dim: Number of output classes (final prediction dimension)
        """
        super(GNN, self).__init__()  # Initialize the parent class (torch.nn.Module)

        # First graph convolutional layer
        # Transforms node features from input_dim to hidden_dim dimensions
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # Second graph convolutional layer
        # Transforms node features from hidden_dim to output_dim dimensions
        self.conv2 = GCNConv(hidden_dim, output_dim)


    def forward(self, data):
        """
        Forward pass through the network.

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node feature matrix with shape [num_nodes, input_dim]
                - edge_index: Graph connectivity in COO format with shape [2, num_edges]
                  where each column [src, dst] represents an edge

        Returns:
            Log probabilities for each class per node
        """
        # Extract node features and the graph structure
        x, edge_index = data.x, data.edge_index

        # First convolution layer:
        # For each node, aggregate features from its neighbors following the GCN formula
        x = self.conv1(x, edge_index)

        # Apply ReLU activation to introduce non-linearity
        x = F.relu(x)

        # Apply dropout for regularization (only active during training)
        # Randomly zeroes some elements to prevent overfitting
        x = F.dropout(x, training=self.training)

        # Second convolution layer:
        # Further transform node features using neighbor information
        x = self.conv2(x, edge_index)

        return x
    

class GNNRegression(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(GNNRegression, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        # For regression, output raw continuous values
        return x  # shape: [num_nodes, output_dim]
    

class GraphSAGE_Classification(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.bn2 = BatchNorm(hidden_dim // 2)

        self.conv3 = SAGEConv(hidden_dim // 2, output_dim)  # output_dim=2 for binary, >2 for multiclass

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        return x  # raw logits


class GraphSAGE_Regression(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.bn2 = BatchNorm(hidden_dim // 2)

        self.conv3 = SAGEConv(hidden_dim // 2, 1)  # Output single value for regression

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        return x.view(-1)  # flatten to shape [num_nodes]

class ImprovedGraphSAGEReg(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = LayerNorm(hidden_dim)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.bn2 = LayerNorm(hidden_dim // 2)

        self.conv3 = SAGEConv(hidden_dim // 2, 1)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x.view(-1)


##Anomaly Detection
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