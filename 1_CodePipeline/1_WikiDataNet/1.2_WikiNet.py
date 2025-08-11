import polars as pl
import graph_tool.all as gt
import numpy as np
from sklearn.utils.extmath import randomized_svd
import scipy as sp
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
from scipy.sparse.linalg import eigsh


def log_progress(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

start_time = time.time()
log_progress(f"Starting pipeline - graph-tool version: {gt.__version__}")

### Loading
log_progress("[1/8] Opening Features...")

df_article_index = pl.read_parquet("data/article_TitleIndex_dict.parquet")
print(f"Loaded {len(df_article_index)} article indices")

df_wikidata_articles = pl.read_parquet("data/wikidata_ready4net.parquet")
print(f"Loaded {len(df_wikidata_articles)} wikidata articles")

### Edges 
log_progress("[2/8] Processing Edges...")

df_edges = pl.read_csv('data/cleaned_edges_final.csv')
print(f"Loaded {len(df_edges)} edges")

title_to_id = dict(zip(df_article_index["title"], df_article_index["id"]))
df_edges = df_edges.with_columns([
    pl.col("source").replace(title_to_id),
    pl.col("target").replace(title_to_id)
]).drop_nulls()
print(f"After mapping and filtering: {len(df_edges)} edges")

list_edges = list(zip(df_edges["source"], df_edges["target"]))

### Graph-tool Network
log_progress("[3/8] Creating Network...")

G_tool = gt.Graph(directed=True)
G_tool.add_edge_list(list_edges)
print(f"Created graph with {G_tool.num_vertices()} vertices and {G_tool.num_edges()} edges")

vertices = list(G_tool.get_vertices())
vertex_array = np.array(vertices)

### Network Metrics (Parallelized)
log_progress("[4/8] Calculating Network Metrics...")

def calculate_pagerank():
    log_progress("  Computing Pagerank...")
    return gt.pagerank(G_tool, epsilon=1e-4, max_iter=1000)

def calculate_katz():
    log_progress("  Computing Katz...")
    return gt.katz(G_tool, alpha=0.01, max_iter=1000, epsilon=1e-6)

def calculate_hits():
    log_progress("  Computing HITS...")
    return gt.hits(G_tool, epsilon=1e-4, max_iter=1000)

def calculate_clustering():
    log_progress("  Computing Clustering...")
    return gt.local_clustering(G_tool)

def calculate_core_numbers():
    log_progress("  Computing Core Numbers...")
    return gt.kcore_decomposition(G_tool)

def calculate_betweenness():
    print("Betweenness (approximated)")
    return gt.betweenness(G_tool, norm=False)

def calculate_node_reciprocity():
    print("Reciprocity")
    A = gt.adjacency(G_tool)
    mutual_matrix = A.multiply(A.T)
    out_degrees = np.array(A.sum(axis=1)).flatten()
    in_degrees = np.array(A.sum(axis=0)).flatten()
    
    mutual_out = np.array(mutual_matrix.sum(axis=1)).flatten()
    max_degrees = np.maximum(in_degrees, out_degrees)
    
    return np.divide(mutual_out, max_degrees, 
                    out=np.zeros_like(mutual_out, dtype=float), 
                    where=max_degrees!=0)

# Calculate degrees once (used by multiple metrics)
in_degrees = G_tool.get_in_degrees(vertices)
out_degrees = G_tool.get_out_degrees(vertices)

# Parallelize independent calculations
with ThreadPoolExecutor(max_workers=2) as executor:
    pagerank_future = executor.submit(calculate_pagerank)
    katz_future = executor.submit(calculate_katz)
    hits_future = executor.submit(calculate_hits)
    clustering_future = executor.submit(calculate_clustering)
    core_future = executor.submit(calculate_core_numbers)
    #betweenness_future = executor.submit(calculate_betweenness)
    #reciprocity_future = executor.submit(calculate_node_reciprocity)
    
    # Collect results
    pagerank = pagerank_future.result()
    katz = katz_future.result()
    hits = hits_future.result()
    clustering = clustering_future.result()
    core_numbers = core_future.result()
    #reciprocity = reciprocity_future.result()
    #betweenness = betweenness_future.result()

eigenvalue, authority_scores, hub_scores = hits  # HITS returns (eigenvalue, authority, hub)

df_wikinetmetrics = pl.DataFrame({
    'pageid': df_wikidata_articles['pageid'],
    'degree_in_centrality': in_degrees,
    'degree_out_centrality': out_degrees,
    'pagerank': pagerank.a,
    'katz': katz.a,
    'hub': hub_scores.a,
    'authority': authority_scores.a,
    'clustering': clustering.a,
    'core_numbers': core_numbers.a,
    #'reciprocity': reciprocity,
    #'betweenness': betweenness.a,
})

### Spectral Embedding
log_progress("[5/8] Computing Spectral Embedding...")

adj_sparse = gt.adjacency(G_tool, weight=None)
print("Making adjacency matrix symmetric...")
adj_symmetric = adj_sparse + adj_sparse.T

print("Computing normalized Laplacian...")
degrees = np.array(adj_symmetric.sum(axis=1)).flatten()
degrees = np.maximum(degrees, 1e-8)
D_inv_sqrt = sp.sparse.diags(1.0 / np.sqrt(degrees), format='csr')
L_norm = sp.sparse.eye(adj_symmetric.shape[0]) - D_inv_sqrt @ adj_symmetric @ D_inv_sqrt

print("Computing smallest eigenvalues...")
eigenvals, eigenvecs = eigsh(L_norm, k=9, which='SM', maxiter=500)
embedding_full = eigenvecs[:, 1:]  

df_wikinetmetrics = df_wikinetmetrics.with_columns([
    pl.Series("spectral_embedding_1", embedding_full[:, 0]),
    pl.Series("spectral_embedding_2", embedding_full[:, 1]), 
    pl.Series("spectral_embedding_3", embedding_full[:, 2]),
    pl.Series("spectral_embedding_4", embedding_full[:, 3]),
    pl.Series("spectral_embedding_5", embedding_full[:, 4]),
    pl.Series("spectral_embedding_6", embedding_full[:, 5]),
    pl.Series("spectral_embedding_7", embedding_full[:, 6]),
    pl.Series("spectral_embedding_8", embedding_full[:, 7]),
])

def compute_transition_features_efficiently(G):
    entropy_features = []
    max_prob_features = []
    concentration_features = []
    
    for v in G.vertices():
        out_degree = v.out_degree()
        
        if out_degree > 0:
            # Uniform transition probabilities (each neighbor equally likely)
            prob = 1.0 / out_degree
            
            # Entropy: -sum(p * log(p))
            entropy = -out_degree * prob * np.log(prob)
            
            # Max probability (same for all in uniform case)
            max_prob = prob
            
            # Concentration: sum(p^2) - higher means more concentrated
            concentration = out_degree * (prob ** 2)
            
        else:
            entropy = 0
            max_prob = 0
            concentration = 0
            
        entropy_features.append(entropy)
        max_prob_features.append(max_prob)
        concentration_features.append(concentration)
    
    return entropy_features, max_prob_features, concentration_features

# Compute all features
entropy_feat, max_prob_feat, concentration_feat = compute_transition_features_efficiently(G_tool)

# Add to dataframe
df_wikinetmetrics = df_wikinetmetrics.with_columns([
    pl.Series("transition_entropy", entropy_feat),
    pl.Series("transition_max_prob", max_prob_feat),
    pl.Series("transition_concentration", concentration_feat)
])

def compute_modularity_features_efficiently(G):
    diagonal_features = []
    row_sum_features = []
    positive_count_features = []
    
    # Get all degrees once
    vertices = list(G.vertices())
    out_degrees = G.get_out_degrees(vertices)
    in_degrees = G.get_in_degrees(vertices)
    total_edges = G.num_edges()
    
    for i, v in enumerate(vertices):
        k_i_out = out_degrees[i]
        k_i_in = in_degrees[i]
        
        # Modularity diagonal: A_ii - (k_i_out * k_i_in) / (2m)
        # A_ii is 0 for simple graphs (no self-loops)
        has_self_loop = 0  # Check if vertex has self-loop
        for e in v.out_edges():
            if e.target() == v:
                has_self_loop = 1
                break
                
        diagonal = has_self_loop - (k_i_out * k_i_in) / (2 * total_edges)
        
        # Row sum calculation: sum over all j of [A_ij - (k_i_out * k_j_in) / (2m)]
        # = degree_out - (k_i_out * sum(k_j_in)) / (2m)
        # = degree_out - (k_i_out * total_in_degree) / (2m)
        # = degree_out - (k_i_out * total_edges) / (2m)  [since sum of in-degrees = total edges]
        # = degree_out - k_i_out / 2
        row_sum = k_i_out - (k_i_out * total_edges) / (2 * total_edges)
        row_sum = k_i_out - k_i_out / 2  # Simplifies to k_i_out / 2
        
        # Positive connections: count neighbors j where A_ij - (k_i * k_j) / (2m) > 0
        positive_count = 0
        for e in v.out_edges():
            j = int(e.target())
            k_j_in = in_degrees[j]
            expected = (k_i_out * k_j_in) / (2 * total_edges)
            actual = 1  # There's an edge
            if actual > expected:
                positive_count += 1
                
        diagonal_features.append(diagonal)
        row_sum_features.append(row_sum)
        positive_count_features.append(positive_count)
    
    return diagonal_features, row_sum_features, positive_count_features

# Compute modularity features
mod_diagonal, mod_row_sum, mod_positive = compute_modularity_features_efficiently(G_tool)

# Add to dataframe
df_wikinetmetrics = df_wikinetmetrics.with_columns([
    pl.Series("modularity_row_sum", mod_row_sum), 
    pl.Series("modularity_positive_connections", mod_positive)
])


### Remove isolated nodes (degree < 1)
log_progress("[6/8] Filtering Network...")
degree_mask = G_tool.get_total_degrees(vertices) >= 2
print(f"Keeping {np.sum(degree_mask)} nodes out of {len(degree_mask)} (degree >= 2)")
G_filt = gt.GraphView(G_tool, vfilt=degree_mask)
G_filt = gt.Graph(G_filt, prune=True)
print(f"Filtered graph: {G_filt.num_vertices()} vertices, {G_filt.num_edges()} edges")

### Merging and Saving
log_progress("[7/8] Merging and Saving...")
df_wiki_fullfeatures = df_wikidata_articles.join(df_wikinetmetrics, how="inner", on="pageid")

# Add all dataframe columns as vertex properties
log_progress("[8/8] Adding vertex properties to graph...")

# Create pageid to vertex mapping for filtered graph
vertex_pageids = {v: None for v in G_filt.vertices()}
for v in G_filt.vertices():
    vertex_pageids[v] = int(v)  # Use vertex index as pageid for now

# Create pageid mapping from original vertices to filtered vertices
original_vertices = list(G_tool.get_vertices())
filtered_vertices = list(G_filt.get_vertices())

# Map pageids from dataframe to filtered graph vertices
pageid_to_vertex = {}
df_pageids = df_wiki_fullfeatures['pageid'].to_list()

for i, v in enumerate(filtered_vertices):
    if i < len(df_pageids):
        pageid_to_vertex[df_pageids[i]] = v

# Add each column as vertex property
for col in df_wiki_fullfeatures.columns:
    if col == 'pageid':
        continue
        
    col_data = df_wiki_fullfeatures[col].to_list()
    
    # Determine property type
    if df_wiki_fullfeatures[col].dtype in [pl.Float32, pl.Float64]:
        v_prop = G_filt.new_vertex_property("double")
    elif df_wiki_fullfeatures[col].dtype in [pl.Int32, pl.Int64]:
        v_prop = G_filt.new_vertex_property("int")
    else:
        v_prop = G_filt.new_vertex_property("string")
    
    # Assign values to vertices
    for i, v in enumerate(filtered_vertices):
        if i < len(col_data):
            if df_wiki_fullfeatures[col].dtype in [pl.Float32, pl.Float64]:
                v_prop[v] = float(col_data[i]) if col_data[i] is not None else 0.0
            elif df_wiki_fullfeatures[col].dtype in [pl.Int32, pl.Int64]:
                v_prop[v] = int(col_data[i]) if col_data[i] is not None else 0
            else:
                v_prop[v] = str(col_data[i]) if col_data[i] is not None else ""
        else:
            v_prop[v] = 0.0 if df_wiki_fullfeatures[col].dtype in [pl.Float32, pl.Float64] else (0 if df_wiki_fullfeatures[col].dtype in [pl.Int32, pl.Int64] else "")
    
    G_filt.vertex_properties[col] = v_prop

print(f"Added {len(df_wiki_fullfeatures.columns)-1} vertex properties")

# Save dataframe
output_path = "data/df_wiki_data.parquet"
df_wiki_fullfeatures.write_parquet(output_path)
print(f"Saved dataframe to: {output_path}")

# Save graph
output_path = "data/G_wiki.gt"
G_filt.save(output_path)
print(f"Saved graph to: {output_path}")
print(f"Graph vertex properties: {list(G_filt.vertex_properties.keys())}")

total_time = time.time() - start_time
log_progress(f"Pipeline completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
