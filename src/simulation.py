import numpy as np
import networkx as nx

class PotentialOutcomeModel:
    """
    Simulates Potential Outcomes based on a 2-hop linear interference model.
    Ref: Equation (17) and (18) in the paper.
    """
    def __init__(self, graph: nx.Graph, avg_deg: float):
        self.g = graph
        self.num_nodes = graph.number_of_nodes()
        self.avg_deg = avg_deg
        
        # Precompute adjacency matrices
        self._precompute_matrices()
        # Precompute covariates for bias generation
        self._precompute_covariates()

    def _precompute_matrices(self):
        """Computes D^-1 A and (D^-1 A)^2 for interference simulation."""
        A = np.array(nx.adjacency_matrix(self.g).todense(), dtype=np.float64)
        deg_array = np.array([self.g.degree[n] for n in self.g.nodes])
        
        # 1-hop normalized adjacency: D^-1 A
        self.D_inv_A = np.zeros_like(A)
        for i in range(self.num_nodes):
            if deg_array[i] > 0:
                self.D_inv_A[i] = A[i] / deg_array[i]
        
        # 2-hop normalized adjacency: (D^-1 A)^2
        self.multi_hop_A = np.linalg.matrix_power(self.D_inv_A, 2)
        # Set diagonal to 0
        np.fill_diagonal(self.multi_hop_A, 0)

    def _precompute_covariates(self):
        """Prepares covariates used in the interaction term."""
        node_list = list(self.g.nodes)
        
        # Covariate 1: Normalized Degree
        degs = np.array([self.g.degree[n] for n in node_list])
        self.norm_deg = degs / self.avg_deg
        
        # Covariate 2: Number of connected clusters (computed in clustering step)
        n_cl = np.array([self.g.nodes[n]["n_cl"] for n in node_list])
        self.norm_n_cl = n_cl / n_cl.mean()

    def generate_outcomes(self, z_vec, alpha=0, beta=1, sigma=2.0, gamma=1, r1=1, r2=0, cov_weight=1, single_covariate=False):
        """
        Generates outcome Y based on treatment Z.
        
        Args:
            single_covariate (bool): If True, uses only 'normalized connected clusters' as covariate.
                                     (Appendix A.2 setting).
        """
        
        if single_covariate:
            # Case discussed in Appendix A.2: Single covariate (normalized connected clusters)
            interaction = cov_weight * self.norm_n_cl.reshape(-1, 1) * z_vec
        else:
            # Default Case: Multiple covariates (Degree + Connected Clusters)
            interaction = cov_weight * (self.norm_deg.reshape(-1, 1) + self.norm_n_cl.reshape(-1, 1)) * z_vec
        
        # Interference Term: r1 * A*z + r2 * A^2*z
        interference = gamma * (r1 * np.matmul(self.D_inv_A, z_vec) + r2 * np.matmul(self.multi_hop_A, z_vec))
        
        # Total Outcome
        g_vec = alpha + beta * z_vec + interaction + interference
        
        # Add Gaussian Noise
        noise = sigma * np.random.normal(size=(self.num_nodes, 1))
        y_vec = g_vec + noise
        
        # Store in graph attributes
        node_list = list(self.g.nodes)
        for i, node in enumerate(node_list):
            self.g.nodes[node]["y"] = y_vec[i][0]
            self.g.nodes[node]["z"] = z_vec[i][0]
            
        return y_vec