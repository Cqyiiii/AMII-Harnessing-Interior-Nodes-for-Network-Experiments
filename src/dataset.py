import pandas as pd
import networkx as nx
import dgl
import numpy as np

def load_graph_data(path: str):
    """
    Loads graph data from an .mtx file and preprocesses it.
    
    Args:
        path (str): Path to the .mtx file.
        
    Returns:
        g (nx.Graph): NetworkX graph object.
        dgl_g (dgl.DGLGraph): DGL graph object.
        avg_deg (float): Average degree of the graph.
    """
    print(f"Loading data from {path}...")
    # Read mtx file (skipping header)
    df = pd.read_table(path, skiprows=1, names=["source", "target"], sep=" ")
    g = nx.from_pandas_edgelist(df)
    
    # Calculate basic statistics
    num_nodes = g.number_of_nodes()
    degs = [g.degree[i] for i in g.nodes]
    avg_deg = sum(degs) / len(degs)
    
    # Store degree as node attribute
    for i in g.nodes:
        g.nodes[i]["deg"] = g.degree[i]
        
    # Convert to DGL graph for GCN training
    dgl_g = dgl.from_networkx(g)
    
    print(f"Graph loaded: {num_nodes} nodes, {g.number_of_edges()} edges.")
    return g, dgl_g, avg_deg