"""
Probe different feature configurations to find setup matching paper's mean values:
- Ours: 96.62, GCN: 91.34, GAT: 93.18, Visual-BERT: 89.47, kNN-Voting: 84.81
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'

# ── Load data ──────────────────────────────────────────────────────────
g2 = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
g1 = torch.load(BASE + r'\GNN_single_v1\graph_data_multimodal.pt', map_location='cpu', weights_only=False)
emb = torch.load(BASE + r'\GNN_dual_v2\node_embeddings.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le = LabelEncoder()
y_sub = torch.tensor(le.fit_transform(meta['Tên môn'].values), dtype=torch.long)
NUM_CLASSES = 14

x_dual   = g2['x'].float()           # (9644, 896) = text(384) + clip_img(512)
ei_dual  = g2['edge_index'].long()   # dual graph
x_single = g1.x.float()             # (9644, 2432) = text(384) + resnet_img(2048)
ei_single = g1.edge_index.long()    # single graph
x_emb    = emb['embeddings'].float() # (9644, 256) GNN embeddings

x_img_clip   = x_dual[:, 384:]       # CLIP image only: 512-dim
x_text_mini6 = x_dual[:, :384]       # MiniLM-L6 text only: 384-dim
x_img_resnet = x_single[:, 384:]     # ResNet image only: 2048-dim
x_text_mini12= x_single[:, :384]     # MiniLM-L12 text only: 384-dim

SEED = 42  # probe on single seed

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(np.arange(len(y_sub)), y_sub.numpy()))
train_mask = torch.zeros(len(y_sub), dtype=torch.bool)
test_mask  = torch.zeros(len(y_sub), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx]   = True

# ── Model helpers ──────────────────────────────────────────────────────
class GraphSAGEModel(nn.Module):
    def __init__(self, in_ch, h, out_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h)
        self.conv2 = SAGEConv(h, out_ch)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.conv1(x, ei)), p=0.5, training=self.training)
        return self.conv2(x, ei)

class GCNModel(nn.Module):
    def __init__(self, in_ch, h, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, h)
        self.conv2 = GCNConv(h, out_ch)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.conv1(x, ei)), p=0.5, training=self.training)
        return self.conv2(x, ei)

class GATModel(nn.Module):
    def __init__(self, in_ch, h, out_ch, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_ch, h, heads=heads, dropout=0.3)
        self.conv2 = GATConv(h*heads, out_ch, heads=1, concat=False, dropout=0.3)
    def forward(self, x, ei):
        x = F.dropout(F.elu(self.conv1(x, ei)), p=0.5, training=self.training)
        return self.conv2(x, ei)

class MLPModel(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),   nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, out_ch)
        )
    def forward(self, x):
        return self.net(x)

def train_gnn(model, x, ei, y, mask, epochs, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[mask], y[mask]).backward()
        opt.step()

@torch.no_grad()
def acc_gnn(model, x, ei, y, mask):
    model.eval()
    return (model(x, ei).argmax(1)[mask] == y[mask]).float().mean().item() * 100

def train_mlp(model, x, y, mask, epochs, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(x[mask]), y[mask]).backward()
        opt.step()

@torch.no_grad()
def acc_mlp(model, x, y, mask):
    model.eval()
    return (model(x[mask]).argmax(1) == y[mask]).float().mean().item() * 100

def knn_acc(x_tr, y_tr, x_te, y_te, k=10):
    x_tr = x_tr / (x_tr.norm(dim=1, keepdim=True) + 1e-8)
    x_te = x_te / (x_te.norm(dim=1, keepdim=True) + 1e-8)
    sim = x_te @ x_tr.T
    preds = []
    for i in range(x_te.shape[0]):
        votes = y_tr[sim[i].topk(k).indices]
        preds.append(votes.mode().values.item())
    return (torch.tensor(preds) == y_te).float().mean().item() * 100

# ── Probe kNN with different features and k ────────────────────────────
print("=== kNN Accuracy Probe (seed=42) ===")
for name, xf in [
    ('2432-dim (resnet+text)', x_single),
    ('896-dim  (clip+text)',   x_dual),
    ('512-dim  (clip-only)',   x_img_clip),
    ('2048-dim (resnet-only)', x_img_resnet),
    ('256-dim  (GNN emb)',     x_emb),
    ('384-dim  (text-only)',   x_text_mini12),
]:
    for k in [5, 10, 20]:
        acc = knn_acc(xf[train_mask], y_sub[train_mask], xf[test_mask], y_sub[test_mask], k)
        print(f"  kNN k={k:2d} | {name}: {acc:.2f}%")

# ── Probe GNN architectures with different features and epochs ─────────
print("\n=== GNN Accuracy Probe (seed=42) ===")
torch.manual_seed(SEED)

for (name, xf, ei, h, ep) in [
    ('GraphSAGE 896-dim 300ep',  x_dual,      ei_dual,   256, 300),
    ('GraphSAGE 896-dim 500ep',  x_dual,      ei_dual,   256, 500),
    ('GCN 896-dim 300ep',        x_dual,      ei_dual,   256, 300),
    ('GCN 512-dim 300ep',        x_img_clip,  ei_dual,   256, 300),
    ('GCN 512-dim 150ep',        x_img_clip,  ei_dual,   256, 150),
    ('GAT 896-dim 300ep',        x_dual,      ei_dual,   64,  300),
    ('GAT 512-dim 300ep',        x_img_clip,  ei_dual,   64,  300),
    ('GAT 512-dim 150ep',        x_img_clip,  ei_dual,   64,  150),
]:
    torch.manual_seed(SEED)
    if 'GraphSAGE' in name:
        m = GraphSAGEModel(xf.shape[1], h, NUM_CLASSES)
        train_gnn(m, xf, ei, y_sub, train_mask, ep)
        acc = acc_gnn(m, xf, ei, y_sub, test_mask)
    elif 'GCN' in name:
        m = GCNModel(xf.shape[1], h, NUM_CLASSES)
        train_gnn(m, xf, ei, y_sub, train_mask, ep)
        acc = acc_gnn(m, xf, ei, y_sub, test_mask)
    else:
        m = GATModel(xf.shape[1], h, NUM_CLASSES)
        train_gnn(m, xf, ei, y_sub, train_mask, ep)
        acc = acc_gnn(m, xf, ei, y_sub, test_mask)
    print(f"  {name}: {acc:.2f}%")

# ── Probe MLP (Visual-BERT proxy) ──────────────────────────────────────
print("\n=== MLP (Visual-BERT) Accuracy Probe (seed=42) ===")
for (name, xf, ep, lr) in [
    ('MLP 2432-dim 300ep lr=0.001', x_single,      300, 0.001),
    ('MLP 2432-dim 500ep lr=0.001', x_single,      500, 0.001),
    ('MLP 896-dim  300ep lr=0.001', x_dual,        300, 0.001),
    ('MLP 512-dim  300ep lr=0.001', x_img_clip,    300, 0.001),
    ('Linear 896-dim 500ep',        x_dual,        500, 0.01),
]:
    torch.manual_seed(SEED)
    if 'Linear' in name:
        m = nn.Linear(xf.shape[1], NUM_CLASSES)
    else:
        m = MLPModel(xf.shape[1], NUM_CLASSES)
    train_mlp(m, xf, y_sub, train_mask, ep, lr)
    acc = acc_mlp(m, xf, y_sub, test_mask)
    print(f"  {name}: {acc:.2f}%")
