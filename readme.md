# Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference

[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the paper **"Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference"**, accepted at **ICLR 2026**.



## 📌 Abstract

A/B testing on platforms often faces challenges from network interference, where a unit's outcome depends on the treatments of its neighbors. We propose the **Mean-in-Interior (MII)** estimator, which leverages "interior nodes" (nodes whose neighbors all belong to the same cluster) to avoid complex reweighting and reduce variance.

To address the potential selection bias of interior nodes (e.g., degree distribution discrepancy), we further introduce the **Augmented MII (AMII)** estimator. AMII incorporates a counterfactual predictor (e.g., a GNN) trained on the full graph to correct the bias, embodying a semi-supervised learning perspective.



## 📂 Project Structure

The code is organized to ensure modularity and reproducibility:

```text
AMII-ICLR2026/
├── Dataset/                 # Directory for graph datasets (e.g., .mtx files)
├── src/                     # Core implementation modules
│   ├── dataset.py           # Graph data loading and preprocessing
│   ├── clustering.py        # Louvain clustering and Interior/Boundary identification
│   ├── simulation.py        # Potential Outcome Model (Linear 2-hop interference)
│   ├── models.py            # GCN architecture for counterfactual prediction
│   └── estimators.py        # Implementations of MII, AMII, CAE, and Hajek estimators
├── main.py                  # Main entry point for running simulations
├── requirements.txt         # Python dependencies
└── README.md                # Project documentationxxxxxxxxxx pip install -r requirements.txtbash
```



## 🚀 Getting Started

### 1. Prerequisites

This project requires Python 3.9+ and the following libraries:

- PyTorch
- DGL (Deep Graph Library)
- NetworkX
- NumPy, Pandas, Scipy

Install them using:

Bash

```
pip install -r requirements.txt
```

### 2. Run Simulation

To reproduce the experimental results (e.g., statistical performance under different treatment proportions), run the main script:

Bash

```
python main.py
```

You can modify configurations (e.g., interference level `R2`, treatment proportions `p`) directly in `main.py` or use command line arguments as follows:

| Argument              | Description                                         | Default |
| --------------------- | --------------------------------------------------- | ------- |
| `--res_param`         | Louvain resolution parameter                        | 5       |
| `--r2`                | Interference level (0: 1-hop, 1: 2-hop)             | 0       |
| `--cov_weight`        | Covariate weight (bias magnitude)                   | 0.5     |
| `--train_on_boundary` | **(Appendix A.3)** Train GNN only on boundary nodes | `False` |
| `--single_covariate`  | **(Appendix A.2)** Use only `n_cl` covariate        | `False` |





## 🖊️ Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{chen2026journey,
  title={Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference},
  author={Chen, Qianyi and Wu, Anpeng and Li, Bo and Deng, Lu and Wang, Yong},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

