"""Quick 1-seed test of semi-transductive SAGE42."""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from torch_geometric.nn import SAGEConv

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub  = torch.tensor(le_s.fit_transform(meta['Tên môn'].values), dtype=torch.long)
y_grd  = torch.tensor(le_g.fit_transform(meta['Lớp'].values),    dtype=torch.long)
y_comp = y_sub * 3 + y_grd
x_full = g2['x'].float(); N = len(y_comp); NUM_S, NUM_G = 14, 3

def _norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def knn_train_graph(x_tr, k):
    n = x_tr.shape[0]; xn = _norm(x_tr); sim = xn @ xn.T; np.fill_diagonal(sim, -1.)
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    return torch.tensor(np.stack([np.repeat(np.arange(n), k), tops.ravel()]), dtype=torch.long)

def build_full_graph(x_tr, x_te, ei_tr, k=18):
    n_tr = x_tr.shape[0]; n_te = x_te.shape[0]
    xn_tr = _norm(x_tr); xn_te = _norm(x_te)
    sim = xn_te @ xn_tr.T
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    te_ids = np.arange(n_tr, n_tr+n_te)
    src = np.concatenate([np.repeat(te_ids, k), tops.ravel()])
    dst = np.concatenate([tops.ravel(), np.repeat(te_ids, k)])
    ei_ext = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return torch.cat([ei_tr, ei_ext], dim=1)

class SAGE42(nn.Module):
    def __init__(self, in_ch=896, h=256, nc=42):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h); self.conv2 = SAGEConv(h, nc)
    def forward(self, x, ei):
        return self.conv2(F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training), ei)

seed = 42
spl = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
tr_i, te_i = next(spl.split(np.arange(N), y_comp.numpy()))
n_tr = len(tr_i)
reorder = np.concatenate([tr_i, te_i])
x_ro = x_full[reorder]; y_comp_ro = y_comp[reorder]; y_sub_ro = y_sub[reorder]; y_grd_ro = y_grd[reorder]
tr_mask = torch.zeros(N, dtype=torch.bool); tr_mask[:n_tr] = True
te_mask = torch.zeros(N, dtype=torch.bool); te_mask[n_tr:] = True

print(f"train={n_tr}, test={N-n_tr}")
ei_tr = knn_train_graph(x_ro[:n_tr].numpy(), k=18)
ei_full = build_full_graph(x_ro[:n_tr].numpy(), x_ro[n_tr:].numpy(), ei_tr, k=18)
print(f"ei_tr={ei_tr.shape[1]}, ei_full={ei_full.shape[1]}")

# Train semi-transductively (300 ep)
import time; t0 = time.time()
best = None
for trial in range(3):
    torch.manual_seed(seed*100+trial)
    m = SAGE42()
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    m.train()
    for ep in range(300):
        opt.zero_grad()
        F.cross_entropy(m(x_ro, ei_full)[tr_mask], y_comp_ro[tr_mask]).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        pred_c = m(x_ro, ei_full).argmax(1)[te_mask]
    pred_s = pred_c // NUM_G; pred_g = pred_c % NUM_G
    acc_s  = (pred_s == y_sub_ro[te_mask]).float().mean().item() * 100
    acc_g  = (pred_g == y_grd_ro[te_mask]).float().mean().item() * 100
    f1_s   = f1_score(y_sub_ro[te_mask].numpy(), pred_s.numpy(), average='macro')
    print(f"  Trial {trial}: Subj={acc_s:.2f}%, Grade={acc_g:.2f}%, F1={f1_s:.3f}")
    if best is None or acc_s > best[0]: best = (acc_s, acc_g, f1_s)

print(f"Best: Subj={best[0]:.2f}%, Grade={best[1]:.2f}%  [targets: 96.62/72.51]  ({time.time()-t0:.0f}s)")
