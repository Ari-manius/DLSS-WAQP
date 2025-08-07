import torch
import numpy as np
from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import StandardScaler


def add_node_degree_features(data, include_centrality=True):
    """
    Add node degree and centrality features to existing node features.
    
    Args:
        data: PyTorch Geometric Data object
        include_centrality: Whether to include centrality measures (slower but more informative)
    
    Returns:
        Updated Data object with enhanced node features
    """
    # Calculate basic degree features
    row, col = data.edge_index
    
    # Node degrees
    in_degree = degree(col, data.num_nodes, dtype=torch.float).view(-1, 1)
    out_degree = degree(row, data.num_nodes, dtype=torch.float).view(-1, 1)
    total_degree = in_degree + out_degree
    
    # Degree ratios (avoid division by zero)
    in_out_ratio = torch.where(out_degree > 0, in_degree / out_degree, torch.zeros_like(in_degree))
    
    # Degree features
    degree_features = torch.cat([in_degree, out_degree, total_degree, in_out_ratio], dim=1)
    
    additional_features = [degree_features]
    feature_names = ['in_degree', 'out_degree', 'total_degree', 'in_out_ratio']
    
    if include_centrality:
        print("Computing centrality measures (this may take a while)...")
        
        # Convert to NetworkX for centrality calculations
        G = to_networkx(data, to_undirected=False)
        
        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G)
            betweenness_values = torch.tensor([betweenness.get(i, 0.0) for i in range(data.num_nodes)], 
                                            dtype=torch.float).view(-1, 1)
            additional_features.append(betweenness_values)
            feature_names.append('betweenness_centrality')
        except:
            print("Warning: Could not compute betweenness centrality")
        
        # Closeness centrality
        try:
            closeness = nx.closeness_centrality(G)
            closeness_values = torch.tensor([closeness.get(i, 0.0) for i in range(data.num_nodes)], 
                                          dtype=torch.float).view(-1, 1)
            additional_features.append(closeness_values)
            feature_names.append('closeness_centrality')
        except:
            print("Warning: Could not compute closeness centrality")
        
        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
            eigenvector_values = torch.tensor([eigenvector.get(i, 0.0) for i in range(data.num_nodes)], 
                                            dtype=torch.float).view(-1, 1)
            additional_features.append(eigenvector_values)
            feature_names.append('eigenvector_centrality')
        except:
            print("Warning: Could not compute eigenvector centrality")
    
    # Combine all additional features
    new_features = torch.cat(additional_features, dim=1)
    
    # Normalize the new features
    scaler = StandardScaler()
    new_features_normalized = torch.tensor(
        scaler.fit_transform(new_features.cpu().numpy()), 
        dtype=torch.float,
        device=data.x.device if data.x is not None else 'cpu'
    )
    
    # Combine with existing features
    if data.x is not None:
        enhanced_features = torch.cat([data.x, new_features_normalized], dim=1)
    else:
        enhanced_features = new_features_normalized
    
    # Create new data object
    enhanced_data = Data(
        x=enhanced_features,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
        y=data.y,
        num_nodes=data.num_nodes
    )
    
    # Copy other attributes
    for key, value in data.__dict__.items():
        if key not in ['x', 'edge_index', 'edge_attr', 'y', 'num_nodes']:
            setattr(enhanced_data, key, value)
    
    print(f"Added {new_features.shape[1]} new features: {feature_names}")
    print(f"Total features: {enhanced_features.shape[1]}")
    
    return enhanced_data, feature_names


def add_local_clustering(data):
    """
    Add local clustering coefficient as a feature.
    
    Args:
        data: PyTorch Geometric Data object
    
    Returns:
        Updated Data object with clustering coefficient feature
    """
    G = to_networkx(data, to_undirected=True)  # Use undirected for clustering
    clustering = nx.clustering(G)
    
    clustering_values = torch.tensor(
        [clustering.get(i, 0.0) for i in range(data.num_nodes)], 
        dtype=torch.float
    ).view(-1, 1)
    
    if data.x is not None:
        enhanced_features = torch.cat([data.x, clustering_values], dim=1)
    else:
        enhanced_features = clustering_values
    
    # Create new data object
    enhanced_data = Data(
        x=enhanced_features,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
        y=data.y,
        num_nodes=data.num_nodes
    )
    
    # Copy other attributes
    for key, value in data.__dict__.items():
        if key not in ['x', 'edge_index', 'edge_attr', 'y', 'num_nodes']:
            setattr(enhanced_data, key, value)
    
    return enhanced_data


def compute_graph_statistics(data):
    """
    Compute and print graph statistics for analysis.
    
    Args:
        data: PyTorch Geometric Data object
    """
    print(f"Graph Statistics:")
    print(f"  Number of nodes: {data.num_nodes}")
    print(f"  Number of edges: {data.edge_index.shape[1]}")
    print(f"  Number of features: {data.num_node_features}")
    
    if data.y is not None:
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        print(f"  Number of classes: {len(unique_labels)}")
        print(f"  Class distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        
        # Calculate class imbalance ratio
        max_count = counts.max().item()
        min_count = counts.min().item()
        imbalance_ratio = max_count / min_count
        print(f"  Class imbalance ratio: {imbalance_ratio:.2f}")
    
    # Degree statistics
    row, col = data.edge_index
    degrees = degree(row, data.num_nodes)
    print(f"  Average degree: {degrees.float().mean().item():.2f}")
    print(f"  Max degree: {degrees.max().item()}")
    print(f"  Min degree: {degrees.min().item()}")


def enhance_existing_data_files():
    """
    Enhance existing data files with new features.
    """
    import os
    
    data_dir = "data"
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    for data_file in data_files:
        print(f"\nEnhancing {data_file}...")
        data_path = os.path.join(data_dir, data_file)
        
        # Load data
        data = torch.load(data_path, weights_only=False)
        
        # Print original statistics
        print("Original:")
        compute_graph_statistics(data)
        
        # Add features
        enhanced_data, feature_names = add_node_degree_features(data, include_centrality=False)
        
        # Print enhanced statistics
        print("Enhanced:")
        compute_graph_statistics(enhanced_data)
        
        # Save enhanced data
        enhanced_filename = data_file.replace('.pt', '_enhanced.pt')
        enhanced_path = os.path.join(data_dir, enhanced_filename)
        torch.save(enhanced_data, enhanced_path)
        
        print(f"Saved enhanced data to {enhanced_filename}")


if __name__ == "__main__":
    enhance_existing_data_files()