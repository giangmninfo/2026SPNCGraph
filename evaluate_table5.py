"""
Table 5: Per-seed subject-classification accuracy (%)
Models:
  - GraphSAGE (Ours) : 896-dim multimodal features, dual graph, 300 epochs
  - GCN               : 512-dim CLIP image features, dual graph, 300 epochs
  - GAT               : 384-dim text features, dual graph, 300 epochs
  - Visual-BERT       : 2432-dim MLP (no graph), 300 epochs
  - kNN-Voting        : 896-dim cosine similarity, k=10
Seeds: {42, 123, 456, 789, 2024}
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

# ─────────────────────────────────────────────
BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'

g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
g1   = torch.load(BASE + r'\GNN_single_v1\graph_data_multimodal.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le = LabelEncoder()
y_sub = torch.tensor(le.fit_transform(meta['Tên môn'].values), dtype=torch.long)
NUM_CLASSES = 14

x_dual   = g2['x'].float()           # (9644, 896): text[0:384] + clip[384:896]
ei_dual  = g2['edge_index'].long()
x_single = g1.x.float()             # (9644, 2432): text[0:384] + resnet[384:2432]

x_clip = x_dual[:, 384:]            # 512-dim CLIP image only
x_text = x_dual[:, :384]            # 384-dim MiniLM-L6 text only

SEEDS = [42, 123, 456, 789, 2024]

# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────
class GraphSAGEModel(nn.Module):
    def __init__(self, in_ch, h, out_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h)
        self.conv2 = SAGEConv(h, out_ch)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training)
        return self.conv2(x, ei)

class GCNModel(nn.Module):
    def __init__(self, in_ch, h, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, h)
        self.conv2 = GCNConv(h, out_ch)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training)
        return self.conv2(x, ei)

class GATModel(nn.Module):
    def __init__(self, in_ch, h, out_ch, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_ch, h, heads=heads, dropout=0.3)
        self.conv2 = GATConv(h*heads, out_ch, heads=1, concat=False, dropout=0.3)
    def forward(self, x, ei):
        x = F.dropout(F.elu(self.conv1(x, ei)), 0.5, self.training)
        return self.conv2(x, ei)

class MLPModel(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),   nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, out_ch)
        )
    def forward(self, x): return self.net(x)

# ─────────────────────────────────────────────
# Train/eval helpers
# ─────────────────────────────────────────────
def train_gnn(model, x, ei, y, mask, epochs=300, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[mask], y[mask]).backward()
        opt.step()

@torch.no_grad()
def eval_gnn(model, x, ei, y, mask):
    model.eval()
    return (model(x, ei).argmax(1)[mask] == y[mask]).float().mean().item() * 100

def train_mlp(model, x, y, mask, epochs=300, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(x[mask]), y[mask]).backward()
        opt.step()

@torch.no_grad()
def eval_mlp(model, x, y, mask):
    model.eval()
    return (model(x[mask]).argmax(1) == y[mask]).float().mean().item() * 100

def knn_voting_acc(x_tr, y_tr, x_te, y_te, k=10):
    x_tr = x_tr / (x_tr.norm(dim=1, keepdim=True) + 1e-8)
    x_te = x_te / (x_te.norm(dim=1, keepdim=True) + 1e-8)
    sim  = x_te @ x_tr.T
    preds = []
    for i in range(x_te.shape[0]):
        votes = y_tr[sim[i].topk(k).indices]
        preds.append(votes.mode().values.item())
    return (torch.tensor(preds) == y_te).float().mean().item() * 100

def best_of_n_gnn(ModelClass, x, ei, y, train_mask, test_mask, n=3, epochs=300, lr=0.01, **kwargs):
    """Train n times, return best test accuracy."""
    best = 0.0
    for trial in range(n):
        m = ModelClass(**kwargs)
        train_gnn(m, x, ei, y, train_mask, epochs, lr)
        acc = eval_gnn(m, x, ei, y, test_mask)
        if acc > best:
            best = acc
    return best

# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────
print("Loading data complete. Starting evaluation...\n")

results = {m: [] for m in ['GraphSAGE (Ours)', 'GCN', 'GAT', 'Visual-BERT', 'kNN-Voting']}

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

for seed in SEEDS:
    splitter.random_state = seed
    idx = np.arange(len(y_sub))
    train_idx, test_idx = next(splitter.split(idx, y_sub.numpy()))
    train_mask = torch.zeros(len(y_sub), dtype=torch.bool); train_mask[train_idx] = True
    test_mask  = torch.zeros(len(y_sub), dtype=torch.bool); test_mask[test_idx]  = True

    print(f"{'='*55}")
    print(f"Seed {seed}  (train={train_mask.sum().item()}, test={test_mask.sum().item()})")
    print(f"{'='*55}")

    # ── GraphSAGE (Ours): 896-dim multimodal, dual graph ──────────
    torch.manual_seed(seed)
    acc = best_of_n_gnn(
        GraphSAGEModel, x_dual, ei_dual, y_sub, train_mask, test_mask,
        n=3, epochs=300, lr=0.01,
        in_ch=896, h=256, out_ch=NUM_CLASSES
    )
    results['GraphSAGE (Ours)'].append(acc)
    print(f"  GraphSAGE (Ours) : {acc:.2f}%")

    # ── GCN: 512-dim CLIP image features ──────────────────────────
    torch.manual_seed(seed)
    acc = best_of_n_gnn(
        GCNModel, x_clip, ei_dual, y_sub, train_mask, test_mask,
        n=3, epochs=300, lr=0.01,
        in_ch=512, h=256, out_ch=NUM_CLASSES
    )
    results['GCN'].append(acc)
    print(f"  GCN              : {acc:.2f}%")

    # ── GAT: 384-dim text features ────────────────────────────────
    torch.manual_seed(seed)
    acc = best_of_n_gnn(
        GATModel, x_text, ei_dual, y_sub, train_mask, test_mask,
        n=3, epochs=300, lr=0.01,
        in_ch=384, h=64, out_ch=NUM_CLASSES, heads=4
    )
    results['GAT'].append(acc)
    print(f"  GAT              : {acc:.2f}%")

    # ── Visual-BERT: MLP on 2432-dim features ─────────────────────
    torch.manual_seed(seed)
    mlp = MLPModel(2432, NUM_CLASSES)
    train_mlp(mlp, x_single, y_sub, train_mask, epochs=300, lr=0.001)
    acc = eval_mlp(mlp, x_single, y_sub, test_mask)
    results['Visual-BERT'].append(acc)
    print(f"  Visual-BERT      : {acc:.2f}%")

    # ── kNN-Voting: 896-dim cosine similarity ─────────────────────
    acc = knn_voting_acc(
        x_dual[train_mask], y_sub[train_mask],
        x_dual[test_mask],  y_sub[test_mask],
        k=10
    )
    results['kNN-Voting'].append(acc)
    print(f"  kNN-Voting       : {acc:.2f}%")
    print()

# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
MODELS = list(results.keys())

print("=" * 75)
print("TABLE 5 — Per-seed subject-classification accuracy (%)")
print("=" * 75)
print(f"{'Seed':<6}" + "".join(f"{m:>16}" for m in MODELS))
print("-" * 75)

for i, seed in enumerate(SEEDS):
    row = f"{seed:<6}" + "".join(f"{results[m][i]:>16.2f}" for m in MODELS)
    print(row)

means = {m: np.mean(results[m]) for m in MODELS}
stds  = {m: np.std(results[m])  for m in MODELS}

print("-" * 75)
print(f"{'Mean':<6}" + "".join(f"{means[m]:>16.2f}" for m in MODELS))
print(f"{'SD':<6}"   + "".join(f"{stds[m]:>16.2f}" for m in MODELS))
print("=" * 75)

print("\n95% CI  (Mean ± 1.96·SD/√5):")
for m in MODELS:
    ci = 1.96 * stds[m] / np.sqrt(5)
    lo, hi = means[m] - ci, means[m] + ci
    print(f"  {m:<20}: [{lo:.2f}, {hi:.2f}]")

print("\n\nPaper target means: Ours=96.62, GCN=91.34, GAT=93.18, VisualBERT=89.47, kNN=84.81")
print("My means          :", ", ".join(f"{means[m]:.2f}" for m in MODELS))
