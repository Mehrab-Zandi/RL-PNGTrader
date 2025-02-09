# RL-PNGTrader
Based on the paper (Re-)Imag(in)ing Price Trends



That’s a great idea! Instead of giving the agent a sequence of past price bars, you provide multiple views of market structure with different window lengths (e.g., 20, 50, 100, 200 bars).


Great question! Visibility graphs (VG) and CNNs on price images are two different ways to extract features from time-series data. Each has its strengths and weaknesses. Let's compare them:
1. CNN on Price Images
Pros

✅ Captures spatial patterns: CNNs can recognize support/resistance, trends, and other visual structures.
✅ Multi-timeframe analysis: You can stack different time windows as channels.
✅ Leverages transfer learning: Pretrained CNNs can be fine-tuned for financial tasks.
✅ Handles distortions well: Price patterns may shift in position/scale, but CNNs can generalize.
Cons

❌ Computationally expensive: Requires generating images at each step.
❌ Feature extraction is implicit: The CNN learns features, but we don’t explicitly know which ones are important.
❌ May not generalize well: If trained on one market, CNNs may struggle in another (scaling problem).
2. Visibility Graphs (VG)
How They Work

A VG converts a time series into a graph where:

    Each price point is a node.
    Two nodes are connected if they “see” each other without another point blocking their view.

This transforms price data into a graph structure, which can be analyzed with Graph Neural Networks (GNNs) or traditional graph measures.
Pros

✅ Lighter and faster: No need for image generation, just adjacency matrices.
✅ Encodes relationships explicitly: Graph features (degree, clustering, centrality) represent price structure.
✅ Scale-invariant: Unlike CNNs, VGs naturally handle different markets and timeframes.
✅ Resilient to noise: Structural features remain stable despite small fluctuations.
Cons

❌ Loss of some spatial information: VGs capture structure but may miss some shape details.
❌ Not as well-researched: Fewer pre-trained models exist compared to CNNs.
❌ Graph models are still evolving: GNNs are promising but less mature than CNNs.
Which One is Better?

    If you want a model that generalizes across markets & timeframes → Visibility Graphs (better scaling).
    If you want to leverage deep learning advances → CNNs on price images (better feature discovery).
    If computation is a concern → Visibility Graphs (lighter & cleaner).
    If interpretability matters → Visibility Graphs (explicit graph features vs. black-box CNNs).

Hybrid Approach?

    You can combine both:
        Use VG features as additional channels in a CNN.
        Use CNNs on VG adjacency matrices instead of raw price images.

Yes! You can train a PPO (Proximal Policy Optimization) agent using visibility graphs (VGs) as matrices for observations. Instead of feeding raw price data or price images, you convert price series into graph structures and use their adjacency matrices (or other graph-based features) as the agent’s observations.
How to Use Visibility Graphs in PPO

    Convert Price Data to a Visibility Graph
        Construct the adjacency matrix (or Laplacian matrix) from the price time series.
        Compute graph features (e.g., node degree, centrality, clustering coefficients).

    Use Graph Features as PPO Observations
        The agent can receive the adjacency matrix directly (like an image) or a vector of graph statistics (simpler).
        You can feed a sequence of past VGs to provide memory.

    Choose a Neural Network Architecture
        Standard MLP (fully connected network) for a vector of graph statistics.
        CNN (if using adjacency matrices as images).
        Graph Neural Network (GNN) for a true graph-based approach.

Advantages of Using VGs in PPO

✅ Stable features – Visibility graphs are robust to scaling issues across markets.
✅ Computationally efficient – Smaller feature sets compared to CNNs on images.
✅ Captures price structure – Graph-based relationships may be better than raw price changes.
✅ Generalizes better – Graphs can be trained on multiple markets and still work.
