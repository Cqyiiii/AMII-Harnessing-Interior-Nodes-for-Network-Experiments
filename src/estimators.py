import numpy as np

def estimate_mii(graph, interior_nodes):
    """
    Calculates the Mean-in-Interior (MII) estimator.
    """
    y_interior_treated = []
    
    for node in interior_nodes:
        if graph.nodes[node]["z"] == 1:
            y_interior_treated.append(graph.nodes[node]["y"])
            
    if len(y_interior_treated) == 0:
        return 0.0
        
    return np.mean(y_interior_treated)

def estimate_amii(graph, mii_est, global_pred, interior_nodes, node_to_idx):
    """
    Calculates the Augmented MII (AMII) estimator.
    Formula: MII + Global_Mean_Pred - Interior_Treated_Mean_Pred
    """
    # 1. Global Mean Prediction
    global_mean_pred = np.mean(global_pred)
    
    # 2. Local Interior Mean Prediction (Average of f(1) over TREATED interior nodes)
    interior_treated_preds = []
    for node in interior_nodes:
        # Crucial: Check if the node is actually treated in the experiment
        if graph.nodes[node]["z"] == 1:
            idx = node_to_idx[node]
            interior_treated_preds.append(global_pred[idx])
            
    if len(interior_treated_preds) == 0:
        local_mean_pred = 0.0
    else:
        local_mean_pred = np.mean(interior_treated_preds)
        
    # AMII = MII + (Global - Local)
    return mii_est + global_mean_pred - local_mean_pred

def estimate_hajek(graph):
    """Hajek Estimator."""
    ht_numerator = 0.0
    ht_denominator = 0.0
    
    for node in graph.nodes:
        w = graph.nodes[node].get("w_HT", 0)
        y = graph.nodes[node]["y"]
        ht_numerator += y * w
        ht_denominator += w
        
    if ht_denominator == 0:
        return 0.0
    return ht_numerator / ht_denominator

def estimate_cae(graph, clusters):
    """Cluster-Adaptive Estimator (CAE)."""
    cluster_means = []
    
    for cluster in clusters:
        y_vals = []
        for node in cluster:
            node_deg = graph.nodes[node]["deg"]
            neighbor_z_sum = sum([graph.nodes[nbr]["z"] for nbr in graph[node]])
            
            if graph.nodes[node]["z"] == 1 and neighbor_z_sum == node_deg:
                y_vals.append(graph.nodes[node]["y"])
        
        if len(y_vals) > 0:
            cluster_means.append(np.mean(y_vals))
            
    if len(cluster_means) == 0:
        return 0.0
    return np.mean(cluster_means)