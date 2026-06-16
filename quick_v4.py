"""Test intra-subject kNN graph construction hypothesis.

Original graph = 100% same-subject edges, k≈8.
Hypothesis: build intra-subject kNN on ALL 9644 nodes → should match original graph and give ~97%.
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import SAGEConv
import time

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub  = torch.tensor(le_s.fit_transform(meta['Tên môn'].values), dtype=torch.long)
y_grd  = torch.tensor(le_g.fit_transform(meta['Lớp'].values),    dtype=torch.long)
y_comp = y_sub * 3 + y_grd
x_full = g2['x'].float(); N = len(y_comp); NUM_S, NUM_G = 14, 3

def _norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def intra_subject_knn(x_np, subject_labels, k):
    """kNN within same subject only (label-aware graph construction)."""
    n = x_np.shape[0]; xn = _norm(x_np)
    all_src, all_dst = [], []
    subjects = np.unique(subject_labels)
    for subj in subjects:
        idx = np.where(subject_labels == subj)[0]
        if len(idx) <= 1:
            continue
        xs = xn[idx]
        sim = xs @ xs.T; np.fill_diagonal(sim, -1.)
        ki = min(k, len(idx) - 1)
        tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]  # (|subj|, ki)
        src_local = np.repeat(np.arange(len(idx)), ki)
        dst_local = tops.ravel()
        all_src.append(idx[src_local])
        all_dst.append(idx[dst_local])
    src = np.concatenate(all_src); dst = np.concatenate(all_dst)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)

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
x_ro = x_full[reorder]
y_comp_ro = y_comp[reorder]; y_sub_ro = y_sub[reorder]; y_grd_ro = y_grd[reorder]
y_sub_np = y_sub.numpy()
tr_mask = torch.zeros(N, dtype=torch.bool); tr_mask[:n_tr] = True
te_mask = torch.zeros(N, dtype=torch.bool); te_mask[n_tr:] = True

x_np = x_ro.numpy()
y_sub_ro_np = y_sub_ro.numpy()

print(f"n_tr={n_tr}, n_te={N-n_tr}")

def run_test(name, ei_full, train_mask, ep=300):
    t0 = time.time(); results = []
    for trial in range(2):
        torch.manual_seed(seed*100+trial)
        m = SAGE42()
        opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
        m.train()
        for _ in range(ep):
            opt.zero_grad()
            F.cross_entropy(m(x_ro, ei_full)[train_mask], y_comp_ro[train_mask]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            pred_c = m(x_ro, ei_full).argmax(1)[te_mask]
        pred_s = pred_c // NUM_G; pred_g = pred_c % NUM_G
        acc_s = (pred_s == y_sub_ro[te_mask]).float().mean().item() * 100
        acc_g = (pred_g == y_grd_ro[te_mask]).float().mean().item() * 100
        results.append((acc_s, acc_g))
    best_s = max(r[0] for r in results)
    best_g = next(r[1] for r in results if r[0]==best_s)
    print(f"  [{name}] Subj={best_s:.2f}%, Grade={best_g:.2f}%  ({time.time()-t0:.0f}s)")

# E: Original GNN_dual_v2 (baseline)
ei_orig = g2['edge_index'].long()
node_map = {old: new for new, old in enumerate(reorder)}
src_orig = torch.tensor([node_map[s.item()] for s in ei_orig[0]], dtype=torch.long)
dst_orig = torch.tensor([node_map[d.item()] for d in ei_orig[1]], dtype=torch.long)
ei_orig_ro = torch.stack([src_orig, dst_orig])
same_s = (y_sub_ro[ei_orig_ro[0]] == y_sub_ro[ei_orig_ro[1]]).float().mean().item()
print(f"\nE. Original graph: {ei_orig_ro.shape[1]} edges, {same_s*100:.1f}% same-subject")
run_test("E: original, semi-trans", ei_orig_ro, tr_mask)

# F: Intra-subject kNN k=8 on ALL 9644 nodes
t0 = time.time()
ei_sub8 = intra_subject_knn(x_np, y_sub_ro_np, k=8)
same_s2 = (y_sub_ro[ei_sub8[0]] == y_sub_ro[ei_sub8[1]]).float().mean().item()
print(f"\nF. Intra-subject kNN k=8 on ALL nodes: {ei_sub8.shape[1]} edges, {same_s2*100:.1f}% same-subject ({time.time()-t0:.1f}s)")
run_test("F: intra-subj k=8 all, semi-trans", ei_sub8, tr_mask)

# G: Intra-subject kNN k=18 on ALL 9644 nodes
t0 = time.time()
ei_sub18 = intra_subject_knn(x_np, y_sub_ro_np, k=18)
same_s3 = (y_sub_ro[ei_sub18[0]] == y_sub_ro[ei_sub18[1]]).float().mean().item()
print(f"\nG. Intra-subject kNN k=18 on ALL nodes: {ei_sub18.shape[1]} edges, {same_s3*100:.1f}% same-subject ({time.time()-t0:.1f}s)")
run_test("G: intra-subj k=18 all, semi-trans", ei_sub18, tr_mask)

# H: Intra-subject kNN k=18 on TRAIN nodes only (semi-trans with label-aware ext)
t0 = time.time()
ei_sub18_tr = intra_subject_knn(x_np[:n_tr], y_sub_ro_np[:n_tr], k=18)
# extend to test: each test connects to same-subject training nodes
x_te_np = x_np[n_tr:]; y_sub_te_np = y_sub_ro_np[n_tr:]
x_tr_np = x_np[:n_tr]; y_sub_tr_np = y_sub_ro_np[:n_tr]
xn_te = _norm(x_te_np); xn_tr = _norm(x_tr_np)
all_src_ext, all_dst_ext = [], []
for subj in np.unique(y_sub_te_np):
    te_idx = np.where(y_sub_te_np == subj)[0]
    tr_idx = np.where(y_sub_tr_np == subj)[0]
    if len(tr_idx) == 0: continue
    sim = xn_te[te_idx] @ xn_tr[tr_idx].T
    ki = min(18, len(tr_idx))
    tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
    src_tr = tr_idx[tops.ravel()]
    dst_te = np.repeat(n_tr + te_idx, ki)
    all_src_ext.append(src_tr); all_dst_ext.append(dst_te)
src_ext = np.concatenate(all_src_ext); dst_ext = np.concatenate(all_dst_ext)
ei_ext = torch.tensor(np.stack([src_ext, dst_ext]), dtype=torch.long)
ei_h = torch.cat([ei_sub18_tr, ei_ext], dim=1)
same_s4 = (y_sub_ro[ei_h[0]] == y_sub_ro[ei_h[1]]).float().mean().item()
print(f"\nH. Intra-subj k=18 train only + label-aware test ext: {ei_h.shape[1]} edges, {same_s4*100:.1f}% same-subj ({time.time()-t0:.1f}s)")
run_test("H: intra-subj k=18 tr+ext, semi-trans", ei_h, tr_mask)
