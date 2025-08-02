import polars as pl
import graph_tool.all as gt
import numpy as np
from sklearn.utils.extmath import randomized_svd
import scipy as sp
from concurrent.futures import ThreadPoolExecutor

print("graph-tool version:", gt.__version__)

### Loading
print("Opening Features...")

df_article_index = pl.read_parquet("data/article_TitleIndex_dict.parquet")
df_wikidata_articles = pl.read_parquet("data/wikidata_ready4net.parquet")

### Edges 
print("Processing Edges...")

df_edges = pl.read_csv('data/cleaned_edges_final.csv')

title_to_id = dict(zip(df_article_index["title"], df_article_index["id"]))
df_edges = df_edges.with_columns([
    pl.col("source").replace(title_to_id),
    pl.col("target").replace(title_to_id)
]).drop_nulls()

list_edges = list(zip(df_edges["source"], df_edges["target"]))

### Graph-tool Network
print("Creating Network...")

G_tool = gt.Graph(directed=True)
G_tool.add_edge_list(list_edges)

vertices = list(G_tool.get_vertices())
vertex_array = np.array(vertices)

### Network Metrics (Parallelized)
print("Calculating Network Metrics...")

def calculate_pagerank():
    print("Pagerank")
    return gt.pagerank(G_tool, epsilon=1e-4, max_iter=100)

def calculate_katz():
    print("Katz")
    return gt.katz(G_tool, alpha=0.01, max_iter=100, epsilon=1e-6)

def calculate_hits():
    print("Hits")
    return gt.hits(G_tool, epsilon=1e-4, max_iter=100)

def calculate_clustering():
    print("Clustering")
    return gt.local_clustering(G_tool)

def calculate_core_numbers():
    print("Core Number")
    return gt.kcore_decomposition(G_tool)

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
    
    # Collect results
    pagerank = pagerank_future.result()
    katz = katz_future.result()
    hits = hits_future.result()
    clustering = clustering_future.result()
    core_numbers = core_future.result()

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
})

### Spectral Embedding
print("Spectral Embedding...")

adj_sparse = gt.adjacency(G_tool, weight=None)

degrees = np.array(adj_sparse.sum(axis=1)).flatten()
D_inv_sqrt = sp.sparse.diags(1.0 / np.sqrt(degrees + 1e-8), format='csr')
A_norm = D_inv_sqrt @ adj_sparse @ D_inv_sqrt

U, s, Vt = randomized_svd(A_norm, n_components=5, n_iter=10, random_state=42)
embedding_full = U  # Unscaled

df_wikinetmetrics = df_wikinetmetrics.with_columns([
    pl.Series("spectral_embedding_1", embedding_full[:, 0]),
    pl.Series("spectral_embedding_2", embedding_full[:, 1]), 
    pl.Series("spectral_embedding_3", embedding_full[:, 2]),
])

### Remove isolated nodes (degree < 1)
print("Filtering Network...")
degree_mask = G_tool.get_total_degrees(vertices) >= 2
G_filt = gt.GraphView(G_tool, vfilt=degree_mask)
G_filt = gt.Graph(G_filt, prune=True)

### Merging and Saving
print("Merging and Saving...")
df_wiki_fullfeatures = df_wikidata_articles.join(df_wikinetmetrics, how="inner", on="pageid")

# Add all dataframe columns as vertex properties
print("Adding vertex properties to graph...")

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
