- stratified sampling (during Splitting)
- Oversampling (during Training Loop)

- take a look at the loss function

- Anomaly detection methods 

Graph Auto-Encoders (GAE)
- Detect anomalous citation patterns between articles
- Identify structural outliers in the Wikipedia network
- Can extend your existing GCN architecture


⏺ The AnomalyGNN I suggested has several key features that make it special for anomaly detection:

  1. Reconstruction-Based Detection

  Unlike your current GNNs that predict labels, it learns to reconstruct node features:
  - Normal nodes: low reconstruction error
  - Anomalous nodes: high reconstruction error

  2. Unsupervised Learning

  - No need for labeled anomaly data
  - Learns normal patterns from your Wikipedia network
  - Can detect unknown types of anomalies

  3. Graph-Aware Encoding

  z = F.relu(self.encoder(data.x, data.edge_index))  # Uses graph structure
  x_recon = self.decoder(z)  # Reconstructs features
  - Considers both node features AND network structure
  - Detects nodes that don't fit their neighborhood patterns

  4. Anomaly Score Calculation

  anomaly_score = torch.norm(data.x - x_recon, dim=1)  # Per-node error

  5. Applications to Your Project

  - Citation anomalies: Articles with unusual citation patterns
  - Quality manipulation: Articles gaming the quality metrics
  - Content outliers: Articles that don't match their network context
  - Temporal anomalies: Suspicious editing behaviors

  The key advantage is it learns what "normal" Wikipedia articles look like in your network, then flags anything that significantly
  deviates from these learned patterns.