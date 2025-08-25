import torch
from .GNN_model import ImprovedGNN, ResidualGCN, GraphAttentionNet, ResidualGraphSAGE, MLPBaseline

def load_trained_model(checkpoint_path, device='auto'):
    """
    Load a trained model with its configuration from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file (with or without '_with_config.pt')
        device: Device to load the model on ('cpu', 'cuda', 'mps', or 'auto')
               'auto' will use 'mps' for GraphSAINT models, 'cpu' for others
    
    Returns:
        tuple: (model, config_dict)
    """
    
    # Model class mapping
    model_classes = {
        'improved_gnn': ImprovedGNN,
        'residual_gcn': ResidualGCN, 
        'gat': GraphAttentionNet,
        'residual_sage': ResidualGraphSAGE,
        'mlp': MLPBaseline
    }
    
    # Try loading with config first
    config_path = checkpoint_path.replace('.pt', '_with_config.pt') if not checkpoint_path.endswith('_with_config.pt') else checkpoint_path
    
    try:
        # First load to CPU to check config
        checkpoint = torch.load(config_path, map_location='cpu', weights_only=False)
        config = checkpoint['model_config']
        
        # Auto-select device based on training method
        if device == 'auto':
            if config.get('use_graphsaint', False):
                device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
                print(f"🎯 GraphSAINT model detected → Using device: {device}")
            else:
                device = torch.device('cpu')
                print(f"🔧 Regular training detected → Using device: {device}")
        else:
            device = torch.device(device)
            print(f"📱 Manual device selection → Using device: {device}")
        
        # Create model with saved configuration
        model_class = model_classes[config['model_type']]
        
        # Handle optional parameters for different model types
        model_kwargs = {
            'input_dim': config['input_dim'],
            'hidden_dim': config['hidden_dim'], 
            'output_dim': config['output_dim'],
            'num_layers': config['num_layers'],
            'dropout': config['dropout']
        }
        
        # Add model-specific parameters
        if config['model_type'] == 'gat':
            # Try to get heads from config, or infer from saved model structure
            if 'heads' in config:
                model_kwargs['heads'] = config['heads']
            else:
                # Infer heads from saved state dict
                state_dict = checkpoint['model_state_dict']
                if 'convs.0.att_src' in state_dict:
                    # att_src shape is [1, heads, hidden_dim]
                    saved_heads = state_dict['convs.0.att_src'].shape[1]
                    model_kwargs['heads'] = saved_heads
                    print(f"⚠️  GAT heads not in config, inferred from checkpoint: {saved_heads}")
                else:
                    model_kwargs['heads'] = 4  # Final fallback
                    print(f"⚠️  Could not infer GAT heads, using default: 4")
            
        model = model_class(**model_kwargs)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Loaded model from: {config_path}")
        print(f"📋 Model type: {config['model_type']}")
        print(f"📏 Architecture: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, output_dim={config['output_dim']}")
        print(f"🔧 Parameters: num_layers={config['num_layers']}, dropout={config['dropout']}")
        if config.get('use_graphsaint'):
            print(f"🎯 GraphSAINT trained: batch_size={config.get('batch_size')}, walk_length={config.get('walk_length')}")
        
        return model, config
        
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        print("💡 Try running training again with the updated script to generate config files")
        raise
    except KeyError as e:
        print(f"❌ Missing configuration key: {e}")
        print("💡 This checkpoint may be from an older version without config")
        raise


def get_model_info(checkpoint_path):
    """
    Get model configuration info without loading the full model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        dict: Model configuration
    """
    config_path = checkpoint_path.replace('.pt', '_with_config.pt') if not checkpoint_path.endswith('_with_config.pt') else checkpoint_path
    
    try:
        checkpoint = torch.load(config_path, map_location='cpu', weights_only=False)
        return checkpoint['model_config']
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return None