import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np

class GraphClusterer:
    """
    Handles graph clustering (Louvain algorithm) and identifies Interior/Boundary nodes.
    """
    def __init__(self, graph: nx.Graph, resolution: float = 5.0, seed: int = 10):
        self.g = graph
        self.resolution = resolution
        self.seed = seed
        self.clusters = []
        self.node_to_cluster = {}
        self.interior_nodes = []
        self.boundary_nodes = []
        
    def run_clustering(self):
        """Runs Louvain clustering and assigns node roles."""
        print(f"Running Louvain clustering (resolution={self.resolution})...")
        self.clusters = nx_comm.louvain_communities(self.g, seed=self.seed, resolution=self.resolution)
        # Sort clusters by size (descending)
        self.clusters = sorted(self.clusters, key=len, reverse=True)
        
        # Create inverse mapping: node_id -> cluster_id
        num_clusters = len(self.clusters)
        self.node_to_cluster = {
            node: cl for cl in range(num_clusters) for node in self.clusters[cl]
        }
        
        # Identify Interior vs Boundary nodes
        self._identify_interior_nodes()
        # Compute number of connected clusters for each node
        self._compute_connected_clusters()

    def _identify_interior_nodes(self):
        """
        Identifies interior nodes.
        Definition: A node is interior if all its neighbors belong to the same cluster.
        Ref: Equation (2) in the paper.
        """
        self.interior_nodes = []
        all_nodes = list(self.g.nodes)
        
        for node in all_nodes:
            cluster_idx = self.node_to_cluster[node]
            is_interior = True
            for neighbor in self.g[node]:
                if self.node_to_cluster[neighbor] != cluster_idx:
                    is_interior = False
                    break
            if is_interior:
                self.interior_nodes.append(node)
                
        self.boundary_nodes = list(set(all_nodes) - set(self.interior_nodes))

    def _compute_connected_clusters(self):
        """Computes how many distinct clusters each node is connected to."""
        for node in self.g.nodes:
            connected_cls = set()
            # Add own cluster
            connected_cls.add(self.node_to_cluster[node])
            # Add neighbors' clusters
            for neighbor in self.g[node]:
                connected_cls.add(self.node_to_cluster[neighbor])
            
            self.g.nodes[node]["n_cl"] = len(connected_cls)

    def get_masks(self):
        """Returns lists of interior and boundary nodes."""
        return self.interior_nodes, self.boundary_nodes