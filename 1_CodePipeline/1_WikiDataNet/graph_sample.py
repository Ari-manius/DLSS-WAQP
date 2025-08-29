#!/usr/bin/env python3

import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
from scipy import sparse
from scipy.sparse.linalg import eigsh

def log_progress(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_largest_component_sample(graph_path, sample_size=1000, seed=42):
    """Load graph, extract largest connected component, and ensure sampled subgraph remains connected"""
    log_progress("Loading full graph...")
    G_full = gt.load_graph(graph_path)
    
    total_vertices = G_full.num_vertices()
    total_edges = G_full.num_edges()
    log_progress(f"Loaded graph: {total_vertices:,} vertices, {total_edges:,} edges")
    
    # Find largest connected component
    log_progress("Finding largest connected component...")
    comp_labels, hist = gt.label_components(G_full, directed=False)
    largest_comp_id = np.argmax(hist)
    largest_comp_size = hist[largest_comp_id]
    
    log_progress(f"Largest component: {largest_comp_size:,} vertices ({largest_comp_size/total_vertices*100:.1f}%)")
    
    # Create filter for largest component
    largest_comp_filter = G_full.new_vertex_property("bool")
    largest_comp_filter.a = (comp_labels.a == largest_comp_id)
    
    # Extract largest component
    G_largest = gt.GraphView(G_full, vfilt=largest_comp_filter)
    G_largest = gt.Graph(G_largest, prune=True)
    
    log_progress(f"Extracted largest component: {G_largest.num_vertices()} vertices, {G_largest.num_edges()} edges")
    
    # Set random seed for reproducible sampling
    np.random.seed(seed)
    random.seed(seed)
    
    # Use iterative approach to ensure connected sample
    log_progress(f"Creating connected sample of {sample_size} vertices...")
    
    # Start with a random seed vertex
    all_vertices = list(G_largest.vertices())
    if sample_size >= len(all_vertices):
        G_sample = G_largest
        sampled_vertices = all_vertices
        log_progress(f"Using entire largest component ({len(all_vertices)} vertices)")
    else:
        # Start with random seed vertex
        seed_vertex = np.random.choice(all_vertices)
        sampled_vertices = [seed_vertex]
        
        # Grow the sample by adding connected neighbors
        candidates = set()
        for v in sampled_vertices:
            for neighbor in G_largest.get_all_neighbors(v):
                if neighbor not in sampled_vertices:
                    candidates.add(neighbor)
        
        # Keep adding vertices until we reach sample_size or run out of candidates
        while len(sampled_vertices) < sample_size and candidates:
            # Randomly select from candidates
            next_vertex = np.random.choice(list(candidates))
            sampled_vertices.append(next_vertex)
            candidates.discard(next_vertex)
            
            # Add new neighbors to candidates
            for neighbor in G_largest.get_all_neighbors(next_vertex):
                if neighbor not in sampled_vertices:
                    candidates.add(neighbor)
        
        log_progress(f"Grew connected sample to {len(sampled_vertices)} vertices")
        
        # Create vertex filter for sampled vertices
        vertex_filter = G_largest.new_vertex_property("bool")
        vertex_filter.a = False
        for v in sampled_vertices:
            vertex_filter[v] = True
        
        # Create sample subgraph
        G_sample = gt.GraphView(G_largest, vfilt=vertex_filter)
        G_sample = gt.Graph(G_sample, prune=True)
        
        # Verify connectivity of final sample
        final_comp_labels, final_hist = gt.label_components(G_sample, directed=False)
        num_components = len([x for x in final_hist if x > 0])
        
        if num_components > 1:
            log_progress(f"Warning: Sample has {num_components} components, keeping largest")
            # Keep only the largest component of the sample
            final_largest_id = np.argmax(final_hist)
            final_filter = G_sample.new_vertex_property("bool")
            final_filter.a = (final_comp_labels.a == final_largest_id)
            G_sample = gt.GraphView(G_sample, vfilt=final_filter)
            G_sample = gt.Graph(G_sample, prune=True)
    
    # Map back to original vertices for quality extraction
    original_vertices = []
    comp_vertex_mapping = []
    
    # Create mapping of original vertices in largest component
    for orig_v in G_full.vertices():
        if largest_comp_filter[orig_v]:
            comp_vertex_mapping.append(orig_v)
    
    # Map final sample vertices to original vertices
    final_sample_vertices = list(G_sample.vertices())
    for i, sample_v in enumerate(final_sample_vertices):
        if i < len(sampled_vertices):
            # Map through the sampling process
            orig_largest_idx = int(sampled_vertices[i]) if i < len(sampled_vertices) else 0
            if orig_largest_idx < len(comp_vertex_mapping):
                original_vertices.append(comp_vertex_mapping[orig_largest_idx])
    
    log_progress(f"Final connected sample: {G_sample.num_vertices()} vertices, {G_sample.num_edges()} edges")
    
    # Final connectivity check
    final_comp_labels, final_hist = gt.label_components(G_sample, directed=False)
    num_final_components = len([x for x in final_hist if x > 0])
    log_progress(f"Final sample has {num_final_components} connected component(s)")
    
    return G_sample, G_full, original_vertices

def get_quality_colors(G_sample, G_full, sampled_vertices):
    """Extract quality information and prepare colors"""
    quality_full = G_full.vertex_properties['Target_QC_aggcat']
    
    # Extract quality values for sampled vertices
    quality_values = []
    for i, v_orig in enumerate(sampled_vertices):
        if i < G_sample.num_vertices():
            quality_val = quality_full[v_orig]
            quality_values.append(quality_val)
    
    # Create color mapping using grey-blue-red colormap
    color_map = {
        0: '#808080',  # Grey for Low Quality
        1: '#4169E1',  # Blue for Medium Quality  
        2: '#DC143C'   # Red for High Quality
    }
    
    # Prepare node colors
    node_colors = [color_map.get(q, '#808080') for q in quality_values]
    
    return quality_values, node_colors

def spectral_embedding_2d(G):
    """Compute 2D spectral embedding using normalized graph Laplacian"""
    log_progress("Computing spectral embedding...")
    
    # Get adjacency matrix as sparse matrix
    A = gt.adjacency(G, weight=None)
    n = A.shape[0]
    
    # Compute degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    
    # Handle isolated vertices
    degrees = np.maximum(degrees, 1e-12)
    
    # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    L_norm = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Compute smallest eigenvalues and eigenvectors
    try:
        # Get 4 smallest eigenvalues to compute gaps
        eigenvals, eigenvecs = eigsh(L_norm, k=min(4, n-1), which='SM', sigma=0)
        
        # Sort by eigenvalue (should already be sorted)
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        log_progress(f"Eigenvalues: {eigenvals[:4]}")
        
        # Use 2nd and 3rd eigenvectors for 2D coordinates (skip trivial first)
        if eigenvecs.shape[1] >= 3:
            pos_2d = eigenvecs[:, [1, 2]]
        elif eigenvecs.shape[1] >= 2:
            # Fallback: use 1st and 2nd if only 2 available
            pos_2d = eigenvecs[:, [0, 1]]
        else:
            # Fallback: random positions if spectral fails
            log_progress("Warning: Spectral embedding failed, using random positions")
            pos_2d = np.random.randn(n, 2)
        
        # Compute eigenvalue gaps
        gaps = []
        if len(eigenvals) > 1:
            for i in range(len(eigenvals)-1):
                gaps.append(eigenvals[i+1] - eigenvals[i])
        
        largest_gap = max(gaps) if gaps else 0
        log_progress(f"Eigenvalue gaps: {gaps}")
        log_progress(f"Largest eigenvalue gap: {largest_gap:.6f}")
        
        # Scale positions for better visualization
        pos_2d = pos_2d * 10
        
        return pos_2d, largest_gap, eigenvals
        
    except Exception as e:
        log_progress(f"Spectral embedding failed: {e}, using random positions")
        pos_2d = np.random.randn(n, 2)
        return pos_2d, 0, []

def create_clean_visualization(G_sample, node_colors, output_path="network_clean.png", use_spectral=True):
    """Create clean network visualization without text or titles"""
    
    if use_spectral:
        # Use spectral embedding
        pos_2d, eigenvalue_gap, eigenvals = spectral_embedding_2d(G_sample)
        log_progress(f"Using spectral embedding with gap: {eigenvalue_gap:.6f}")
    else:
        # Use force-directed layout
        log_progress("Computing force-directed layout...")
        pos = gt.sfdp_layout(G_sample, K=1., C=0.2, p=1.0, max_iter=500)
        pos_2d = pos.get_2d_array([0, 1]).T
        eigenvalue_gap = None
    
    log_progress("Creating clean visualization...")
    
    # Create figure with no text
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_aspect('equal')
    
    # Draw edges (limited for performance)
    edge_count = 0
    max_edges = 2000
    for e in G_sample.edges():
        if edge_count >= max_edges:
            break
        source_idx = int(e.source())
        target_idx = int(e.target())
        if source_idx < len(pos_2d) and target_idx < len(pos_2d):
            ax.plot([pos_2d[source_idx][0], pos_2d[target_idx][0]], 
                   [pos_2d[source_idx][1], pos_2d[target_idx][1]], 
                   'k-', alpha=0.1, linewidth=0.3, zorder=1)
            edge_count += 1
    
    log_progress(f"Drew {edge_count} edges")
    
    # Draw nodes
    ax.scatter(pos_2d[:, 0], pos_2d[:, 1], 
               c=node_colors, s=20, alpha=0.8, zorder=2, edgecolors='white', linewidth=0.2)
    
    log_progress(f"Drew {len(pos_2d)} nodes")
    
    # Remove all text, labels, ticks, spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove any padding/margins
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save clean visualization
    log_progress(f"Saving clean visualization to {output_path}")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0,
                facecolor='white', edgecolor='none')
    
    plt.show()
    return fig, eigenvalue_gap

def main():
    """Main execution function"""
    log_progress("Starting largest connected component visualization...")
    
    # Configuration
    GRAPH_PATH = "data/G_wiki.gt"
    SAMPLE_SIZE = 2048
    OUTPUT_IMAGE = "wiki_network_clean.png"
    
    try:
        # Load largest component and sample
        G_sample, G_full, sampled_vertices = load_largest_component_sample(
            GRAPH_PATH, SAMPLE_SIZE
        )
        
        # Get quality colors
        quality_values, node_colors = get_quality_colors(
            G_sample, G_full, sampled_vertices
        )
        
        # Create clean visualization with spectral embedding
        _, eigenvalue_gap = create_clean_visualization(G_sample, node_colors, OUTPUT_IMAGE, use_spectral=True)
        
        log_progress("Spectral visualization completed!")
        print(f"✓ Spectral network saved as: {OUTPUT_IMAGE}")
        if eigenvalue_gap:
            print(f"✓ Eigenvalue gap: {eigenvalue_gap:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the graph file 'data/G_wiki.gt' exists.")

if __name__ == "__main__":
    main()