import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import load_graph_data
from src.clustering import GraphClusterer
from src.simulation import PotentialOutcomeModel
from src.models import GCNPredictor
# Import the updated estimator function
from src.estimators import estimate_mii, estimate_amii, estimate_hajek, estimate_cae

# --- Default Configuration ---
DEFAULT_RES_PARAM = 5            
DEFAULT_R2 = 0                   
DEFAULT_COV_WEIGHT = 0.5         

DATA_PATH = 'Dataset/socfb-Stanford3.mtx'
REPEAT_NUM = 100         
TREATMENT_PROPS = [0.1, 0.3, 0.5] 
SIGMA = 2                
RESULT_DIR = 'results'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AMII Simulation")
    
    # 1. Parameter Overrides
    parser.add_argument('--res_param', type=float, default=DEFAULT_RES_PARAM, 
                        help='Louvain resolution parameter')
    parser.add_argument('--r2', type=int, default=DEFAULT_R2, choices=[0, 1],
                        help='Interference level (0 or 1)')
    parser.add_argument('--cov_weight', type=float, default=DEFAULT_COV_WEIGHT, 
                        help='Covariate weight')
    
    # 2. Ablation Settings
    parser.add_argument('--train_on_boundary', action='store_true',
                        help='If set, train GNN only on boundary nodes (Appendix A.3)')
    parser.add_argument('--single_covariate', action='store_true',
                        help='If set, use only normalized connected clusters as covariate (Appendix A.2)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    RES_PARAM = args.res_param
    R2 = args.r2
    COV_WEIGHT = args.cov_weight
    
    print(f"--- Configuration ---")
    print(f"Resolution: {RES_PARAM}, R2: {R2}, Cov Weight: {COV_WEIGHT}")
    print(f"Train on Boundary: {args.train_on_boundary}")
    print(f"Single Covariate Mode: {args.single_covariate}")
    print(f"---------------------")

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    g, dgl_g, avg_deg = load_graph_data(DATA_PATH)
    
    clusterer = GraphClusterer(g, resolution=RES_PARAM)
    clusterer.run_clustering()
    interior_nodes, boundary_nodes = clusterer.get_masks()
    
    node_list = list(g.nodes)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    boundary_indices = torch.tensor([node_to_idx[n] for n in boundary_nodes], dtype=torch.long)
    
    po_model = PotentialOutcomeModel(g, avg_deg)
    
    true_gate = 2 + COV_WEIGHT * 2 if R2 == 0 else 2.978 + COV_WEIGHT * 2
    print(f"True GATE: {true_gate:.3f}")

    for p in TREATMENT_PROPS:
        print(f"\n--- Simulation for Treatment Proportion p={p} ---")
        
        bias_results = {'Hajek': [], 'CAE': [], 'MII': [], 'GNN': [], 'AMII': []}
        
        for seed in tqdm(range(REPEAT_NUM), desc=f"Simulating p={p}"):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            num_clusters = len(clusterer.clusters)
            rollout = np.random.uniform(0, 1, size=(num_clusters,))
            treated_cluster_indices = np.where(rollout < p)[0]
            
            z_vec = np.zeros((g.number_of_nodes(), 1))
            for c_idx in treated_cluster_indices:
                for node in clusterer.clusters[c_idx]:
                    z_vec[node_to_idx[node]] = 1
                    
            po_model.generate_outcomes(
                z_vec, 
                sigma=SIGMA, 
                r2=R2, 
                cov_weight=COV_WEIGHT,
                single_covariate=args.single_covariate
            )
            
            for node in g.nodes:
                n_cl = g.nodes[node]["n_cl"]
                g.nodes[node]["w_HT"] = (1/p) ** n_cl if g.nodes[node]["z"] == 1 else 0

            features = torch.tensor([[g.nodes[n]['z'], g.nodes[n]['deg']] for n in node_list], dtype=torch.float)
            y_target = torch.tensor([[g.nodes[n]['y']] for n in node_list], dtype=torch.float).reshape(-1)
            
            model = GCNPredictor()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            
            model.train()
            for _ in range(200): 
                optimizer.zero_grad()
                pred = model(dgl_g, features).squeeze()
                
                if args.train_on_boundary:
                    loss = F.mse_loss(pred[boundary_indices], y_target[boundary_indices])
                else:
                    loss = F.mse_loss(pred, y_target)
                
                loss.backward()
                optimizer.step()
                
            feat_global_treat = features.clone()
            feat_global_treat[:, 0] = 1 
            
            model.eval()
            with torch.no_grad():
                global_pred_1 = model(dgl_g, feat_global_treat).detach().numpy().flatten()
            
            mii_val = estimate_mii(g, interior_nodes)
            
            # --- FIX IS HERE: Added 'g' as the first argument ---
            amii_val = estimate_amii(g, mii_val, global_pred_1, interior_nodes, node_to_idx)
            
            hajek_val = estimate_hajek(g)
            cae_val = estimate_cae(g, clusterer.clusters)
            gnn_val = np.mean(global_pred_1)
            
            bias_results['MII'].append(mii_val - true_gate)
            bias_results['AMII'].append(amii_val - true_gate)
            bias_results['Hajek'].append(hajek_val - true_gate)
            bias_results['CAE'].append(cae_val - true_gate)
            bias_results['GNN'].append(gnn_val - true_gate)

        HT_array = np.array(bias_results['Hajek'])
        CAE_array = np.array(bias_results['CAE'])
        MII_array = np.array(bias_results['MII'])
        GNN_array = np.array(bias_results['GNN'])
        PPI_array = np.array(bias_results['AMII']) 
        
        estm_list = [HT_array, CAE_array, MII_array, GNN_array, PPI_array]
        
        bias_list = [x.mean() for x in estm_list]
        std_list = [x.std() for x in estm_list]
        mse_list = [x.mean()**2 + x.var() for x in estm_list]
        
        print(f"Bias\t Hajek: {bias_list[0]:.3f}\t CAE: {bias_list[1]:.3f}\t MII: {bias_list[2]:.3f}\t GNN: {bias_list[3]:.3f}\t AMII: {bias_list[4]:.3f}")
        print(f"Std \t Hajek: {std_list[0]:.3f}\t CAE: {std_list[1]:.3f}\t MII: {std_list[2]:.3f}\t GNN: {std_list[3]:.3f}\t AMII: {std_list[4]:.3f}")
        print(f"MSE \t Hajek: {mse_list[0]:.3f}\t CAE: {mse_list[1]:.3f}\t MII: {mse_list[2]:.3f}\t GNN: {mse_list[3]:.3f}\t AMII: {mse_list[4]:.3f}")
        
        suffix = ""
        if args.train_on_boundary:
            suffix += "_bndTrain"
        if args.single_covariate:
            suffix += "_singleCov"

        file_name = f"estm_list_res{RES_PARAM}_p{p}_cov{COV_WEIGHT}_r2{R2}{suffix}.pkl"
        save_path = os.path.join(RESULT_DIR, file_name)
        
        torch.save(estm_list, save_path)
        print(f"Saved results to: {save_path}")

if __name__ == '__main__':
    main()