import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv, GATv2Conv

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
g2  = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
g1  = torch.load(BASE + r'\GNN_single_v1\graph_data_multimodal.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')
le  = LabelEncoder()
y_sub = torch.tensor(le.fit_transform(meta['Tên môn'].values), dtype=torch.long)
NUM_CLASSES = 14

x_dual   = g2['x'].float()
ei_dual  = g2['edge_index'].long()
x_single = g1.x.float()
ei_single= g1.edge_index.long()
x_img_clip = x_dual[:, 384:]   # 512-dim

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(np.arange(len(y_sub)), y_sub.numpy()))
train_mask = torch.zeros(len(y_sub), dtype=torch.bool); train_mask[train_idx] = True
test_mask  = torch.zeros(len(y_sub), dtype=torch.bool); test_mask[test_idx]  = True

def train_eval(model, x, ei, mask_tr, mask_te, epochs, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[mask_tr], y_sub[mask_tr]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return (model(x, ei).argmax(1)[mask_te] == y_sub[mask_te]).float().mean().item()*100

class GAT_std(nn.Module):
    def __init__(self, in_ch, h, out_ch, heads=4, drop=0.5):
        super().__init__()
        self.drop = drop
        self.conv1 = GATConv(in_ch, h, heads=heads, dropout=0.3)
        self.conv2 = GATConv(h*heads, out_ch, heads=1, concat=False, dropout=0.3)
    def forward(self, x, ei):
        x = F.dropout(F.elu(self.conv1(x, ei)), p=self.drop, training=self.training)
        return self.conv2(x, ei)

class GAT_v2(nn.Module):
    def __init__(self, in_ch, h, out_ch, heads=4, drop=0.5):
        super().__init__()
        self.drop = drop
        self.conv1 = GATv2Conv(in_ch, h, heads=heads, dropout=0.3)
        self.conv2 = GATv2Conv(h*heads, out_ch, heads=1, concat=False, dropout=0.3)
    def forward(self, x, ei):
        x = F.dropout(F.elu(self.conv1(x, ei)), p=self.drop, training=self.training)
        return self.conv2(x, ei)

print("=== Looking for GAT config giving ~93% ===")

for (label, xf, ei, Model, h, heads, drop, ep, lr) in [
    # Standard GAT with various feature dims
    ("GAT-std 512-dim h=128 ep=500", x_img_clip, ei_dual,   GAT_std, 128, 4, 0.5, 500, 0.01),
    ("GAT-std 512-dim h=64  ep=500", x_img_clip, ei_dual,   GAT_std,  64, 4, 0.5, 500, 0.01),
    # GATv2 with clip features
    ("GATv2   512-dim h=64  ep=300", x_img_clip, ei_dual,   GAT_v2,   64, 4, 0.5, 300, 0.01),
    ("GATv2   512-dim h=64  ep=500", x_img_clip, ei_dual,   GAT_v2,   64, 4, 0.5, 500, 0.01),
    # GAT on single graph with clip features
    ("GAT-std 512-dim single-graph ep=300", x_img_clip, ei_single, GAT_std, 64, 4, 0.5, 300, 0.01),
    # GAT 896-dim with very few epochs
    ("GAT-std 896-dim h=64  ep=15",  x_dual,     ei_dual,   GAT_std,  64, 4, 0.5,  15, 0.01),
    ("GAT-std 896-dim h=64  ep=20",  x_dual,     ei_dual,   GAT_std,  64, 4, 0.5,  20, 0.01),
    ("GAT-std 896-dim h=64  ep=25",  x_dual,     ei_dual,   GAT_std,  64, 4, 0.5,  25, 0.01),
    # text-only features 384-dim
    ("GAT-std 384-text ep=300",      x_dual[:,:384], ei_dual, GAT_std, 64, 4, 0.5, 300, 0.01),
    # Larger dropout
    ("GAT-std 896-dim drop=0.8 ep=300", x_dual,  ei_dual,   GAT_std,  64, 4, 0.8, 300, 0.01),
    # 1 attention head
    ("GAT-std 896-dim 1head ep=300", x_dual,     ei_dual,   GAT_std,  64, 1, 0.5, 300, 0.01),
]:
    torch.manual_seed(42)
    m = Model(xf.shape[1], h, NUM_CLASSES, heads=heads, drop=drop)
    acc = train_eval(m, xf, ei, train_mask, test_mask, ep, lr)
    print(f"  {label}: {acc:.2f}%")
