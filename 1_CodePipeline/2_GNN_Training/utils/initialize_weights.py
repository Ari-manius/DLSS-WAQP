from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch import nn
import torch
import math

def initialize_weights(model, init_type='improved'):
    """
    Enhanced weight initialization for GNN models.
    
    Args:
        model: The model to initialize
        init_type: 'improved' (default), 'kaiming', or 'xavier'
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # Use different strategies based on layer type
            if 'output_proj' in name or 'classifier' in name:
                # Output layers: Xavier for stable gradients
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif 'input_proj' in name:
                # Input layers: Xavier with small gain for stability
                nn.init.xavier_uniform_(m.weight, gain=0.8)
            else:
                # Hidden layers: Choose based on init_type
                if init_type == 'improved':
                    # Combine benefits of both: Xavier with ReLU gain
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                    
            # Better bias initialization
            if m.bias is not None:
                if 'output_proj' in name:
                    nn.init.zeros_(m.bias)  # Zero for output layers
                else:
                    # Small positive bias for hidden layers (helps with dead neurons)
                    nn.init.constant_(m.bias, 0.01)

        elif isinstance(m, GCNConv):
            # Enhanced GCN initialization
            if hasattr(m, 'lin') and m.lin is not None:
                if hasattr(m.lin, 'weight'):
                    # GCN benefits from scaled Xavier initialization
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.lin.weight)
                    std = math.sqrt(2.0 / (fan_in + fan_out))
                    nn.init.normal_(m.lin.weight, 0, std)
                if hasattr(m.lin, 'bias') and m.lin.bias is not None:
                    nn.init.constant_(m.lin.bias, 0.01)
                    
            # Handle different GCN architectures
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight, gain=0.9)

        elif isinstance(m, SAGEConv):
            # SAGE uses separate transformations for self and neighbor features
            if hasattr(m, 'lin_l') and m.lin_l is not None:
                # Self-connection: preserve more information
                nn.init.xavier_uniform_(m.lin_l.weight, gain=1.2)
                if m.lin_l.bias is not None:
                    nn.init.constant_(m.lin_l.bias, 0.01)
                    
            if hasattr(m, 'lin_r') and m.lin_r is not None:
                # Neighbor aggregation: slightly more aggressive
                nn.init.xavier_uniform_(m.lin_r.weight, gain=1.0)
                if m.lin_r.bias is not None:
                    nn.init.constant_(m.lin_r.bias, 0.01)
                    
            # Handle unified linear layer in some SAGE implementations
            if hasattr(m, 'lin') and m.lin is not None:
                nn.init.xavier_uniform_(m.lin.weight, gain=1.1)
                if hasattr(m.lin, 'bias') and m.lin.bias is not None:
                    nn.init.constant_(m.lin.bias, 0.01)
                    
        elif isinstance(m, GATConv):
            # Enhanced GAT initialization for better attention learning
            # Linear transformations for source and destination nodes
            if hasattr(m, 'lin_src') and m.lin_src is not None:
                nn.init.xavier_uniform_(m.lin_src.weight, gain=1.0)
                if hasattr(m.lin_src, 'bias') and m.lin_src.bias is not None:
                    nn.init.constant_(m.lin_src.bias, 0.01)
                    
            if hasattr(m, 'lin_dst') and m.lin_dst is not None:
                nn.init.xavier_uniform_(m.lin_dst.weight, gain=1.0)
                if hasattr(m.lin_dst, 'bias') and m.lin_dst.bias is not None:
                    nn.init.constant_(m.lin_dst.bias, 0.01)
                    
            # Unified linear layer for some GAT implementations
            if hasattr(m, 'lin') and m.lin is not None:
                nn.init.xavier_uniform_(m.lin.weight, gain=1.0)
                if hasattr(m.lin, 'bias') and m.lin.bias is not None:
                    nn.init.constant_(m.lin.bias, 0.01)
            
            # Attention mechanism weights - critical for learning good attention patterns
            if hasattr(m, 'att_src') and m.att_src is not None:
                # Start with small random values for attention diversity
                nn.init.normal_(m.att_src, 0, 0.1)
                
            if hasattr(m, 'att_dst') and m.att_dst is not None:
                nn.init.normal_(m.att_dst, 0, 0.1)
                
            if hasattr(m, 'att') and m.att is not None:
                # Unified attention parameter
                nn.init.normal_(m.att, 0, 0.1)
                
            # Edge attention weights if present
            if hasattr(m, 'att_edge') and m.att_edge is not None:
                nn.init.normal_(m.att_edge, 0, 0.05)
                
            # Final bias for attention output
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

        # Handle normalization layers
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
                
        # Handle embedding layers if present
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.1)
            
    # Additional model-specific initialization
    _apply_model_specific_init(model)


def _apply_model_specific_init(model):
    """Apply model-specific initialization tweaks."""
    model_name = model.__class__.__name__.lower()
    
    if 'gat' in model_name or 'attention' in model_name:
        # For attention models, ensure attention weights start diverse
        for name, param in model.named_parameters():
            if 'att' in name and len(param.shape) >= 1:
                # Add small noise to break symmetry in attention heads
                with torch.no_grad():
                    param.add_(torch.randn_like(param) * 0.01)
                    
    elif 'residual' in model_name:
        # For residual models, initialize the final layers with smaller weights
        for name, module in model.named_modules():
            if 'output_proj' in name and isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.mul_(0.8)  # Slightly reduce output layer magnitude


def initialize_weights_advanced(model, strategy='adaptive'):
    """
    Advanced initialization with different strategies.
    
    Args:
        model: Model to initialize
        strategy: 'adaptive', 'uniform', or 'layer_sequential'
    """
    if strategy == 'adaptive':
        initialize_weights(model, 'improved')
    elif strategy == 'uniform':
        initialize_weights(model, 'xavier')
    elif strategy == 'layer_sequential':
        # Initialize layers with decreasing variance (helps with very deep networks)
        layer_count = 0
        for m in model.modules():
            if isinstance(m, (nn.Linear, GCNConv, SAGEConv, GATConv)):
                layer_count += 1
                
        current_layer = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                current_layer += 1
                scale = 1.0 - 0.1 * (current_layer / max(layer_count, 1))
                nn.init.xavier_uniform_(m.weight, gain=scale)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01 * scale)
    else:
        initialize_weights(model, 'improved')