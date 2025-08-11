from utils.create_split_masks import create_split_masks
from utils.evaluate_gnn_model import evaluate_gnn_model
from utils.evaluate_gnn_model_lazy import evaluate_gnn_model_lazy
import torch 

def model_data_judged_auto(data, check, use_lazy_loading=False, batch_size=1000):
    from utils.model_loader import get_model_info
    from utils.GNN_model import ImprovedGNN, ResidualGCN, GraphAttentionNet, ResidualGraphSAGE
    
    # Load configuration
    config = get_model_info(f'check/{check}.pt')
    
    # Model class mapping
    model_classes = {
        'improved_gnn': ImprovedGNN,
        'residual_gcn': ResidualGCN, 
        'gat': GraphAttentionNet,
        'residual_sage': ResidualGraphSAGE
    }
    
    # Create model with config parameters
    model_class = model_classes[config['model_type']]
    model_kwargs = {
        'input_dim': config['input_dim'],
        'hidden_dim': config['hidden_dim'], 
        'output_dim': config['output_dim'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout']
    }
    
    if config['model_type'] == 'gat':
        model_kwargs['heads'] = config.get('heads', 4)
        
    model = model_class(**model_kwargs)
    
    # Load state dict on CPU first
    checkpoint_path = f'check/{check}.pt'.replace('.pt', '_with_config.pt') if not f'check/{check}.pt'.endswith('_with_config.pt') else f'check/{check}.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to MPS if available
    if use_lazy_loading:
        device = 'mps'
    else:
        device = 'cpu'
    
    model = model.to(device)
    model.eval()
    
    if use_lazy_loading:
        # Use lazy loading evaluation - only pass the data path
        data_path = f"data/{data}.pt"
        result = evaluate_gnn_model_lazy(data_path, model, mask_type='test', device=device, batch_size=batch_size)
    else:
        # Original evaluation method - load full graph
        data_classification = torch.load(f"data/{data}.pt", weights_only=False)
        _, _, test_mask = create_split_masks(data_classification)
        data_classification.test_mask = test_mask
        result = evaluate_gnn_model(data_classification, model, mask_type='test', device=device)
    
    return result, config

result, config = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_residual_gcn_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config['model_type']} with hidden_dim={config['hidden_dim']}")

result, config = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_residual_sage_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config['model_type']} with hidden_dim={config['hidden_dim']}")

result, config = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_improved_gnn_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config['model_type']} with hidden_dim={config['hidden_dim']}")

result, config = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_gat_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config['model_type']} with hidden_dim={config['hidden_dim']}")

result, config = model_data_judged_auto("data_quantile_Target_QC_aggcat", "enhanced_mlp_data_quantile_Target_QC_aggcat", use_lazy_loading=True, batch_size=8192)
print(f"Model used: {config['model_type']} with hidden_dim={config['hidden_dim']}")