import torch
import torch.nn as nn
import dgl.nn

class GCNPredictor(nn.Module):
    """
    Simple GCN for Counterfactual Prediction.
    Used to approximate f(1, X, A) for the AMII estimator.
    """
    def __init__(self, in_feats=2, h_feats=16, out_feats=1):
        super(GCNPredictor, self).__init__()
        # Chebyshev Convolution filters (as used in the paper code)
        self.conv1 = dgl.nn.ChebConv(in_feats, h_feats, k=2)
        self.conv2 = dgl.nn.ChebConv(h_feats, h_feats, k=1)
        self.conv3 = dgl.nn.ChebConv(h_feats, out_feats, k=1)
        
    def forward(self, g, features):
        x = self.conv1(g, features)
        x = torch.relu(x) # Optional: Paper code didn't use relu explicitly in the snippet but usually needed
        x = self.conv2(g, x)
        x = torch.relu(x)
        x = self.conv3(g, x)
        return x