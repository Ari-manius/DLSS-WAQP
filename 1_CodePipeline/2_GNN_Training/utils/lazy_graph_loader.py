import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

class LazyGraphLoader:
    """
    Lazy loading mechanism for large graphs that processes subgraphs on-demand
    without loading the entire graph into memory at once.
    """
    
    def __init__(self, data_path, batch_size=1000, device='mps'):
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device
        self._metadata = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load only the metadata and structure info without full graph data"""
        print("Loading metadata...")
        # Load just the structure without features on CPU first
        data = torch.load(self.data_path, map_location='cpu', weights_only=False)
        print(f"Data loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        
        # Create split masks if they don't exist
        print("Creating split masks...")
        from .create_split_masks import create_split_masks
        if not hasattr(data, 'test_mask') or data.test_mask is None:
            print("Creating test mask...")
            _, _, test_mask = create_split_masks(data)
            data.test_mask = test_mask
        if not hasattr(data, 'val_mask') or data.val_mask is None:
            print("Creating val mask...")
            _, val_mask, _ = create_split_masks(data)
            data.val_mask = val_mask
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            print("Creating train mask...")
            train_mask, _, _ = create_split_masks(data)
            data.train_mask = train_mask
        
        print("Masks created successfully")
        
        self._metadata = {
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'edge_index': data.edge_index,
            'num_features': data.num_features,
            'num_classes': len(torch.unique(data.y)),
            'y': data.y,
            'test_mask': data.test_mask,
            'val_mask': data.val_mask,
            'train_mask': data.train_mask
        }
        # Don't keep the full x features in memory
        del data
    
    def get_subgraph_batch(self, node_indices):
        """
        Load and return a subgraph containing only the specified nodes
        and their k-hop neighborhoods
        """
        # Load full data temporarily on CPU first, then move subgraph to device
        full_data = torch.load(self.data_path, map_location='cpu', weights_only=False)
        
        # Extract subgraph
        subgraph_result = subgraph(
            node_indices,
            full_data.edge_index,
            relabel_nodes=True,
            num_nodes=full_data.num_nodes
        )
        
        # Handle different return formats from different PyG versions
        if len(subgraph_result) == 2:
            edge_index, edge_mask = subgraph_result
        elif len(subgraph_result) == 4:
            subset, edge_index, mapping, edge_mask = subgraph_result
        else:
            raise ValueError(f"Unexpected subgraph return format: {len(subgraph_result)} values")
        
        # Create subgraph data object
        subgraph_data = Data(
            x=full_data.x[node_indices],
            edge_index=edge_index,
            y=full_data.y[node_indices],
            num_nodes=len(node_indices)
        )
        
        # Add masks if they exist
        for mask_name in ['test_mask', 'val_mask', 'train_mask']:
            if hasattr(full_data, mask_name):
                original_mask = getattr(full_data, mask_name)
                if original_mask is not None:
                    setattr(subgraph_data, mask_name, original_mask[node_indices])
        
        # Clean up
        del full_data
        
        return subgraph_data.to(self.device)
    
    def get_test_node_batches(self, mask_type='test'):
        """
        Generator that yields batches of test nodes for evaluation
        """
        if mask_type == 'test':
            mask = self._metadata['test_mask']
        elif mask_type == 'val':
            mask = self._metadata['val_mask']
        else:
            raise ValueError("mask_type must be 'test' or 'val'")
        
        if mask is None:
            raise ValueError(f"No {mask_type} mask found in data")
        
        test_indices = torch.where(mask)[0]
        #print(f"Found {len(test_indices)} {mask_type} nodes to evaluate")
        
        # Split into batches
        for i in range(0, len(test_indices), self.batch_size):
            batch_indices = test_indices[i:i + self.batch_size]
            print(f"Creating batch {i//self.batch_size + 1}: nodes {i} to {min(i + self.batch_size, len(test_indices))}")
            
            # For GNNs, we need k-hop neighbors, so expand the node set
            # Use only 1-hop neighbors to keep subgraph manageable
            #print("Computing k-hop neighbors...")
            expanded_indices = self._get_k_hop_neighbors(batch_indices, k=1)
            #print(f"Expanded from {len(batch_indices)} to {len(expanded_indices)} nodes")
            
            # If expansion is still too large, limit it
            if len(expanded_indices) > 50000:  # Reasonable limit
                #print(f"Subgraph too large ({len(expanded_indices)} nodes), using direct batch only")
                expanded_indices = batch_indices
            
            yield batch_indices, expanded_indices
    
    def _get_k_hop_neighbors(self, node_indices, k=1):
        """Get k-hop neighbors of the given nodes using optimized approach"""
        edge_index = self._metadata['edge_index']
        current_nodes = set(node_indices.tolist())
        
        # Convert edge_index to more efficient format for lookup
        #print(f"Processing k-hop neighbors for k={k}...")
        
        for hop in range(k):
            #print(f"Computing hop {hop + 1}/{k}, current nodes: {len(current_nodes)}")
            new_nodes = set()
            
            # More efficient neighbor finding using boolean indexing
            current_tensor = torch.tensor(list(current_nodes))
            
            # Find all edges where source is in current_nodes
            source_mask = torch.isin(edge_index[0], current_tensor)
            target_neighbors = edge_index[1][source_mask].unique()
            new_nodes.update(target_neighbors.tolist())
            
            # Find all edges where target is in current_nodes  
            target_mask = torch.isin(edge_index[1], current_tensor)
            source_neighbors = edge_index[0][target_mask].unique()
            new_nodes.update(source_neighbors.tolist())
            
            current_nodes.update(new_nodes)
            #print(f"After hop {hop + 1}: {len(current_nodes)} nodes")
        
        return torch.tensor(list(current_nodes), dtype=torch.long)
    
    @property
    def metadata(self):
        return self._metadata