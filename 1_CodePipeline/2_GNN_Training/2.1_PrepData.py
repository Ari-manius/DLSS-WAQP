import torch
import graph_tool as gt
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder

def graphtool_to_pytorch_geometric(
    graph, 
    node_features=None,
    edge_features=None,
    node_labels=None,
    edge_labels=None,
    include_all_vertex_properties=False,
    categorical_encoding='label',
    device=None,
    verbose=True
):
    """
    Convert a graph-tool Graph to PyTorch Geometric Data object.
    
    Args:
        graph: graph-tool Graph object
        node_features: Node features as property name(s) or array/tensor
        edge_features: Edge features as property name(s) or array/tensor  
        node_labels: Node labels as property name or array/tensor
        edge_labels: Edge labels as property name or array/tensor
        include_all_vertex_properties: If True, include all vertex properties as features
        categorical_encoding: How to handle categorical data ('label', 'onehot', 'embedding')
        device: Target device for tensors
        verbose: Whether to print information about the conversion
        
    Returns:
        PyTorch Geometric Data object
    """
    
    # Get number of nodes and edges
    num_nodes = graph.num_vertices()
    num_edges = graph.num_edges()
    
    if verbose:
        print(f"Converting graph with {num_nodes} nodes and {num_edges} edges")
    
    # Extract edge list with improved efficiency
    if num_edges > 0:
        # Use numpy for better performance on large graphs
        edge_array = np.array([[int(e.source()), int(e.target())] for e in graph.edges()])
        edge_index = torch.from_numpy(edge_array.T).contiguous().long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Move to device if specified
    if device:
        edge_index = edge_index.to(device)
    
    # Enhanced node feature extraction
    x = None
    feature_info = {}
    
    if include_all_vertex_properties and graph.vertex_properties:
        x, feature_info = _extract_all_vertex_properties(
            graph, categorical_encoding, device
        )
    elif node_features is not None:
        x = _extract_features(
            graph, node_features, 'vertex', categorical_encoding, device
        )
    
    # Enhanced edge feature extraction
    edge_attr = None
    if edge_features is not None:
        edge_attr = _extract_features(
            graph, edge_features, 'edge', categorical_encoding, device
        )
    
    # Extract labels
    y = _extract_labels(graph, node_labels, 'vertex', device)
    edge_y = _extract_labels(graph, edge_labels, 'edge', device)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        edge_y=edge_y,
        num_nodes=num_nodes
    )
    
    # Add metadata
    if feature_info:
        data.feature_info = feature_info
    
    
    return data

def _extract_all_vertex_properties(graph, categorical_encoding, device):
    """Extract all vertex properties as features with improved handling."""
    feature_list = []
    property_info = {}
    skipped_properties = []
    
    for prop_name, prop in graph.vertex_properties.items():
        try:
            # Sample a few values to determine data type
            sample_values = [prop[v] for v in list(graph.vertices())[:min(10, graph.num_vertices())]]
            
            if all(isinstance(val, (int, float, np.number)) for val in sample_values):
                # Numeric property
                prop_values = torch.tensor([float(prop[v]) for v in graph.vertices()], 
                                         dtype=torch.float)
                if prop_values.dim() == 1:
                    prop_values = prop_values.unsqueeze(1)
                
                feature_list.append(prop_values)
                property_info[prop_name] = {
                    'type': 'numeric',
                    'shape': prop_values.shape[1:],
                    'range': (prop_values.min().item(), prop_values.max().item())
                }
                
            elif all(isinstance(val, bool) for val in sample_values):
                # Boolean property
                prop_values = torch.tensor([float(prop[v]) for v in graph.vertices()], 
                                         dtype=torch.float).unsqueeze(1)
                feature_list.append(prop_values)
                property_info[prop_name] = {'type': 'boolean', 'shape': (1,)}
                
            elif all(isinstance(val, str) for val in sample_values):
                # String/categorical property
                if categorical_encoding == 'label':
                    le = LabelEncoder()
                    string_values = [prop[v] for v in graph.vertices()]
                    encoded_values = le.fit_transform(string_values)
                    prop_values = torch.tensor(encoded_values, dtype=torch.float).unsqueeze(1)
                    
                    feature_list.append(prop_values)
                    property_info[prop_name] = {
                        'type': 'categorical_label',
                        'shape': (1,),
                        'classes': list(le.classes_),
                        'num_classes': len(le.classes_)
                    }
                elif categorical_encoding == 'onehot':
                    le = LabelEncoder()
                    string_values = [prop[v] for v in graph.vertices()]
                    label_encoded = le.fit_transform(string_values)
                    
                    ohe = OneHotEncoder(sparse_output=False)
                    onehot_encoded = ohe.fit_transform(label_encoded.reshape(-1, 1))
                    prop_values = torch.tensor(onehot_encoded, dtype=torch.float)
                    
                    feature_list.append(prop_values)
                    property_info[prop_name] = {
                        'type': 'categorical_onehot',
                        'shape': (len(le.classes_),),
                        'classes': list(le.classes_),
                        'num_classes': len(le.classes_)
                    }
                else:
                    skipped_properties.append(f"{prop_name} (string, no encoding specified)")
                    
        except Exception as e:
            skipped_properties.append(f"{prop_name} (error: {str(e)})")
            continue
    
    
    # Combine all features
    x = None
    if feature_list:
        # Move to device before concatenating
        if device:
            feature_list = [f.to(device) for f in feature_list]
        x = torch.cat(feature_list, dim=1)
        
    
    return x, property_info

def _extract_features(graph, features, prop_type, categorical_encoding, device):
    """Extract features from graph properties or arrays."""
    if isinstance(features, str):
        features = [features]
    
    if isinstance(features, list):
        # Multiple property names
        feature_list = []
        properties = graph.vertex_properties if prop_type == 'vertex' else graph.edge_properties
        elements = graph.vertices() if prop_type == 'vertex' else graph.edges()
        
        for feat_name in features:
            if feat_name not in properties:
                raise ValueError(f"{prop_type.capitalize()} property '{feat_name}' not found")
            
            prop = properties[feat_name]
            feat_tensor = torch.tensor([prop[elem] for elem in elements], dtype=torch.float)
            
            if feat_tensor.dim() == 1:
                feat_tensor = feat_tensor.unsqueeze(1)
            
            feature_list.append(feat_tensor)
        
        result = torch.cat(feature_list, dim=1)
        
    elif isinstance(features, str):
        # Single property name
        properties = graph.vertex_properties if prop_type == 'vertex' else graph.edge_properties
        elements = graph.vertices() if prop_type == 'vertex' else graph.edges()
        
        if features not in properties:
            raise ValueError(f"{prop_type.capitalize()} property '{features}' not found")
        
        prop = properties[features]
        result = torch.tensor([prop[elem] for elem in elements], dtype=torch.float)
        
        if result.dim() == 1:
            result = result.unsqueeze(1)
    else:
        # Array or tensor
        result = torch.tensor(features, dtype=torch.float)
        if result.dim() == 1:
            result = result.unsqueeze(1)
    
    if device:
        result = result.to(device)
    
    return result

def _extract_labels(graph, labels, prop_type, device):
    """Extract labels from graph properties or arrays."""
    if labels is None:
        return None
    
    if isinstance(labels, str):
        properties = graph.vertex_properties if prop_type == 'vertex' else graph.edge_properties
        elements = graph.vertices() if prop_type == 'vertex' else graph.edges()
        
        if labels not in properties:
            raise ValueError(f"{prop_type.capitalize()} property '{labels}' not found")
        
        prop = properties[labels]
        result = torch.tensor([prop[elem] for elem in elements], dtype=torch.long)
    else:
        result = torch.tensor(labels, dtype=torch.long)
    
    if device:
        result = result.to(device)
    
    return result

def filter_and_scale_properties(
    graph,
    target_variable=None,
    scaling_method='standard',
    exclude_patterns=['pageid', 'Target_'],
    include_target_pattern='Target_',
    verbose=True
):
    """
    Filter graph properties to keep one Target_ variable and scale numerical features.
    
    Parameters:
    -----------
    graph : graph-tool Graph
        The input graph
    target_variable : str or None
        Specific Target_ variable to keep as target. If None, uses first Target_ found
    scaling_method : str
        Scaling method: 'standard', 'minmax', or 'robust'
    exclude_patterns : list
        Patterns to exclude from features (like 'pageid', 'Target_')
    include_target_pattern : str
        Pattern for target variables
    verbose : bool
        Whether to print information
        
    Returns:
    --------
    dict with 'features', 'target', 'scaler', 'feature_names'
    """
    vertex_props = graph.vertex_properties
    
    # Find Target_ variables
    target_vars = [name for name in vertex_props.keys() if include_target_pattern in name]
    
    
    # Select target variable
    if target_variable:
        if target_variable not in target_vars:
            raise ValueError(f"Target variable '{target_variable}' not found. Available: {target_vars}")
        selected_target = target_variable
    elif target_vars:
        selected_target = target_vars[0]
    else:
        selected_target = None
    
    # Filter feature properties (exclude pageid and Target_ variables)
    feature_props = []
    for name, prop in vertex_props.items():
        # Skip if matches exclude patterns
        if any(pattern in name.lower() for pattern in [p.lower() for p in exclude_patterns]):
            continue
        # Skip if it's a target variable (but not the selected one)
        if include_target_pattern in name and name != selected_target:
            continue
        feature_props.append(name)
    
    
    # Extract feature data
    feature_data = []
    for prop_name in feature_props:
        prop = vertex_props[prop_name]
        values = np.array([float(prop[v]) for v in graph.vertices()])
        feature_data.append(values)
    
    if feature_data:
        feature_matrix = np.column_stack(feature_data)
        
        # Apply scaling
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("scaling_method must be 'standard', 'minmax', or 'robust'")
        
        scaled_features = scaler.fit_transform(feature_matrix)
        
    else:
        scaled_features = None
        scaler = None
    
    # Extract target data
    target_data = None
    if selected_target:
        target_prop = vertex_props[selected_target]
        target_data = np.array([float(target_prop[v]) for v in graph.vertices()])
    
    return {
        'features': scaled_features,
        'target': target_data,
        'scaler': scaler,
        'feature_names': feature_props,
        'target_name': selected_target
    }

def load_gt_for_pytorch(
    gt_file_path, 
    target_variable=None,
    scaling_method='standard',
    edge_features=None,
    device=None,
    verbose=True
):
    """
    Load .gt file and convert to PyTorch Geometric format with enhanced features.
    
    Parameters:
    -----------
    gt_file_path : str
        Path to .gt file
    target_variable : str or None
        Specific Target_ variable to use as target
    scaling_method : str
        Scaling method: 'standard', 'minmax', or 'robust'
    edge_features : list, str, or None
        List of edge property names for edge features or single property name
    device : torch.device or None
        Target device for tensors
    verbose : bool
        Whether to print conversion information
        
    Returns:
    --------
    torch_geometric.data.Data, dict with scaling info
    """
    
    # Load graph-tool graph
    
    g_gt = gt.load_graph(gt_file_path)
    

    filtered_data = filter_and_scale_properties(
        g_gt,
        target_variable=target_variable,
        scaling_method=scaling_method,
        verbose=verbose
    )
    
    # Convert filtered and scaled data to PyTorch Geometric
    if filtered_data['features'] is not None:
        x = torch.tensor(filtered_data['features'], dtype=torch.float)
        if device:
            x = x.to(device)
    else:
        x = None
        
    if filtered_data['target'] is not None:
        y = torch.tensor(filtered_data['target'], dtype=torch.long)
        if device:
            y = y.to(device)
    else:
        y = None
    
    # Get edge information
    num_nodes = g_gt.num_vertices()
    num_edges = g_gt.num_edges()
    
    if num_edges > 0:
        edge_array = np.array([[int(e.source()), int(e.target())] for e in g_gt.edges()])
        edge_index = torch.from_numpy(edge_array.T).contiguous().long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    if device:
        edge_index = edge_index.to(device)
    
    # Handle edge features if specified
    edge_attr = None
    if edge_features is not None:
        edge_attr = _extract_features(
            g_gt, edge_features, 'edge', 'label', device
        )
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes
    )
    

    torch.save(data, f"data/data_{scaling_method}_{target_variable}.pt")
    
    return data, filtered_data


# Execute the data preparation
data_filtered_1, scaling_info_1 = load_gt_for_pytorch(
    "../1_WikiDataNet/data/G_wiki.gt", 
    target_variable="Target_QC_aggcat",
    scaling_method='minmax',
    verbose=True
)
print(f"Features: {data_filtered_1.x.shape}")
print(f"Feature name: {scaling_info_1['feature_names']}")
print(f"Target name: {scaling_info_1['target_name']}")


data_filtered_2, scaling_info_2 = load_gt_for_pytorch(
    "../1_WikiDataNet/data/G_wiki.gt", 
    target_variable="Target_QC_numlog",
    scaling_method='minmax',
    verbose=True
)
print(f"Features: {data_filtered_2.x.shape}")
print(f"Feature name: {scaling_info_2['feature_names']}")
print(f"Target name: {scaling_info_2['target_name']}")