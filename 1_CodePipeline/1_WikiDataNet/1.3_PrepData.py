import torch
import graph_tool as gt
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, QuantileTransformer, PowerTransformer

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
        Scaling method: 'standard', 'minmax', 'robust', 'quantile', 'power', or 'log_robust'
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
            scaled_features = scaler.fit_transform(feature_matrix)
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
        elif scaling_method == 'robust':
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
        elif scaling_method == 'quantile':
            # Handle NaNs and determine appropriate n_quantiles
            feature_matrix_clean = np.nan_to_num(feature_matrix, nan=0.0)
            
            # Adaptive n_quantiles based on data size and uniqueness
            n_samples = feature_matrix_clean.shape[0]
            n_unique_vals = len(np.unique(feature_matrix_clean.flatten()))
            n_quantiles = min(500, n_unique_vals, n_samples // 2)
            n_quantiles = max(10, n_quantiles)  # Minimum 10 quantiles
            
            if verbose:
                print(f"Using n_quantiles={n_quantiles} for QuantileTransformer")
            
            scaler = QuantileTransformer(
                output_distribution='uniform', 
                n_quantiles=n_quantiles,
                subsample=200000,  # Limit subsample for large datasets
                random_state=42
            )
            scaled_features = scaler.fit_transform(feature_matrix_clean)
            
            # Post-scaling normalization to ensure stable gradients
            scaled_features = (scaled_features - scaled_features.mean(axis=0)) / (scaled_features.std(axis=0) + 1e-8)
            
            # Final NaN check
            if np.isnan(scaled_features).any():
                print("Warning: NaNs detected after quantile scaling, falling back to robust scaling")
                scaler = RobustScaler()
                scaled_features = scaler.fit_transform(feature_matrix_clean)
        elif scaling_method == 'power':
            try:
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                scaled_features = scaler.fit_transform(feature_matrix)
            except Exception as e:
                print(f"PowerTransformer failed: {e}")
                print("Falling back to QuantileTransformer...")
                scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
                scaled_features = scaler.fit_transform(feature_matrix)
        elif scaling_method == 'log_robust':
            # Log transform then robust scaling for positive-skewed data
            log_features = np.log1p(np.maximum(feature_matrix, 0))  # log1p handles zeros, max ensures non-negative
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(log_features)
        else:
            raise ValueError("scaling_method must be 'standard', 'minmax', 'robust', 'quantile', 'power', or 'log_robust'")
        
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
        Scaling method: 'standard', 'minmax', 'robust', 'quantile', 'power', or 'log_robust'
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
    

    # Save PyTorch data
    torch.save(data, f"../2_GNN_Training/data/data_{scaling_method}_{target_variable}.pt")
    
    # Create and save filtered non-network version
    # Load reference parquet to get non-network feature names
    try:
        df_ref = pd.read_parquet("data/wikidata_ready4net.parquet")
        exclude_patterns = ['pageid', 'Target_']
        reference_feature_names = []
        for col in df_ref.columns:
            if any(pattern.lower() in col.lower() for pattern in exclude_patterns):
                continue
            if 'Target_' in col:
                continue
            reference_feature_names.append(col)
        
        # Filter current features to only those in reference
        if filtered_data['features'] is not None and filtered_data['feature_names']:
            available_features = filtered_data['feature_names']
            matching_features = [f for f in reference_feature_names if f in available_features]
            matching_indices = [available_features.index(f) for f in matching_features if f in available_features]
            
            if matching_indices:
                # Create filtered feature matrix
                filtered_features = filtered_data['features'][:, matching_indices]
                filtered_x = torch.tensor(filtered_features, dtype=torch.float)
                if device:
                    filtered_x = filtered_x.to(device)
                
                # Create filtered data object
                data_nonnetwork = Data(
                    x=filtered_x,
                    edge_index=edge_index,
                    edge_attr=None,  # No edge features for non-network version
                    y=y,
                    num_nodes=num_nodes
                )
                
                # Save filtered version
                torch.save(data_nonnetwork, f"../2_GNN_Training/data/data_nonnetwork_{scaling_method}_{target_variable}.pt")
                
                if verbose:
                    print(f"Created non-network version with {len(matching_features)} features: {matching_features}")
                    print(f"Saved to: ../2_GNN_Training/data/data_nonnetwork_{scaling_method}_{target_variable}.pt")
            
    except Exception as e:
        if verbose:
            print(f"Could not create non-network version: {e}")
    
    # Save scaled data as parquet
    if filtered_data['features'] is not None:
        # Create DataFrame with scaled features
        df_scaled = pd.DataFrame(
            filtered_data['features'], 
            columns=filtered_data['feature_names']
        )
        
        # Add target if exists
        if filtered_data['target'] is not None:
            df_scaled[filtered_data['target_name']] = filtered_data['target']
        
        # Add node IDs for reference
        df_scaled['node_id'] = range(len(df_scaled))
        
        # Save as parquet
        parquet_path = f"data/scaled_data_{scaling_method}_{target_variable}.parquet"
        df_scaled.to_parquet(parquet_path, index=False)
        print(f"Saved scaled data to: {parquet_path}")
    
    return data, filtered_data


#Execute the data preparation
# data_filtered_1, scaling_info_1 = load_gt_for_pytorch(
#     "../1_WikiDataNet/data/G_wiki.gt", 
#     target_variable="Target_QC_aggcat",
#     scaling_method='robust',
#     verbose=True
# )
# print(f"Features: {data_filtered_1.x.shape}")
# print(f"Feature name: {scaling_info_1['feature_names']}")
# print(f"Target name: {scaling_info_1['target_name']}")

# # Execute the data preparation
# data_filtered_2, scaling_info_2 = load_gt_for_pytorch(
#     "../1_WikiDataNet/data/G_wiki.gt", 
#     target_variable="Target_QC_aggcat",
#     scaling_method='power',
#     verbose=True
# )
# print(f"Features: {data_filtered_1.x.shape}")
# print(f"Feature name: {scaling_info_1['feature_names']}")
# print(f"Target name: {scaling_info_1['target_name']}")

# # Execute the data preparation
# data_filtered_3, scaling_info_3 = load_gt_for_pytorch(
#     "../1_WikiDataNet/data/G_wiki.gt", 
#     target_variable="Target_QC_aggcat",
#     scaling_method='log_robust',
#     verbose=True
# )
# print(f"Features: {data_filtered_3.x.shape}")
# print(f"Feature name: {scaling_info_3['feature_names']}")
# print(f"Target name: {scaling_info_3['target_name']}")

# Execute the data preparation
data_filtered_4, scaling_info_4 = load_gt_for_pytorch(
    "data/G_wiki.gt", 
    target_variable="Target_QC_aggcat",
    scaling_method='quantile',
    verbose=True
)
print(f"Features: {data_filtered_4.x.shape}")
print(f"Feature name: {scaling_info_4['feature_names']}")
print(f"Target name: {scaling_info_4['target_name']}")

