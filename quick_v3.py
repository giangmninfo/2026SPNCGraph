"""
Test different edge direction strategies for inductive GraphSAGE evaluation.

In SAGEConv: edge (src->dst) means src sends message to dst.
For test node t to aggregate from training neighbors:
  - Need edges (tr_neighbor -> t) i.e. tr→te direction

For training nodes to NOT be influenced by test nodes:
  - Do NOT add (t -> tr_neighbor) edges
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
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

def knn_train_graph(x_tr, k):
    n = x_tr.shape[0]; xn = _norm(x_tr); sim = xn @ xn.T; np.fill_diagonal(sim, -1.)
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    return torch.tensor(np.stack([np.repeat(np.arange(n), k), tops.ravel()]), dtype=torch.long)

def build_graph_tr_to_te(x_tr, x_te, ei_tr, k=18):
    """Add only tr→te edges: training neighbors send to test nodes.
    Training nodes NOT influenced by test nodes (correct inductive setup)."""
    n_tr = x_tr.shape[0]; n_te = x_te.shape[0]
    xn_tr = _norm(x_tr); xn_te = _norm(x_te)
    sim = xn_te @ xn_tr.T
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    te_ids = np.arange(n_tr, n_tr + n_te)
    src_tr = tops.ravel()               # training sends
    dst_te = np.repeat(te_ids, k)        # to test
    ei_ext = torch.tensor(np.stack([src_tr, dst_te]), dtype=torch.long)
    return torch.cat([ei_tr, ei_ext], dim=1)

def build_graph_bidirectional(x_tr, x_te, ei_tr, k=18):
    """Bidirectional edges: training<->test (previous approach)."""
    n_tr = x_tr.shape[0]; n_te = x_te.shape[0]
    xn_tr = _norm(x_tr); xn_te = _norm(x_te)
    sim = xn_te @ xn_tr.T
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    te_ids = np.arange(n_tr, n_tr + n_te)
    src_te = np.repeat(te_ids, k); dst_tr = tops.ravel()
    src = np.concatenate([src_te, dst_tr]); dst = np.concatenate([dst_tr, src_te])
    ei_ext = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return torch.cat([ei_tr, ei_ext], dim=1)

def build_graph_transductive(x_tr, x_te, k=18):
    """Build kNN on ALL nodes (transductive), no train/test distinction."""
    x_all = np.concatenate([x_tr, x_te], axis=0)
    n = x_all.shape[0]; xn = _norm(x_all); sim = xn @ xn.T; np.fill_diagonal(sim, -1.)
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    return torch.tensor(np.stack([np.repeat(np.arange(n), k), tops.ravel()]), dtype=torch.long)

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

x_tr_np = x_ro[:n_tr].numpy(); x_te_np = x_ro[n_tr:].numpy()

print(f"n_tr={n_tr}, n_te={N-n_tr}")

ei_tr = knn_train_graph(x_tr_np, k=18)
print(f"ei_tr shape: {ei_tr.shape}")

def run_test(name, ei_full, train_on_tr_only=True, ep=300):
    t0 = time.time()
    results = []
    for trial in range(2):
        torch.manual_seed(seed*100+trial)
        m = SAGE42()
        opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
        m.train()
        for _ in range(ep):
            opt.zero_grad()
            if train_on_tr_only:
                # train on training subgraph: x_ro[:n_tr] + ei_tr
                F.cross_entropy(m(x_ro[:n_tr], ei_tr), y_comp_ro[:n_tr]).backward()
            else:
                # train on full graph with tr_mask
                F.cross_entropy(m(x_ro, ei_full)[tr_mask], y_comp_ro[tr_mask]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            pred_c = m(x_ro, ei_full).argmax(1)[te_mask]
        pred_s = pred_c // NUM_G; pred_g = pred_c % NUM_G
        acc_s  = (pred_s == y_sub_ro[te_mask]).float().mean().item() * 100
        acc_g  = (pred_g == y_grd_ro[te_mask]).float().mean().item() * 100
        results.append((acc_s, acc_g))
    best_s = max(r[0] for r in results)
    best_g = next(r[1] for r in results if r[0]==best_s)
    print(f"  [{name}] Subj={best_s:.2f}%, Grade={best_g:.2f}%  ({time.time()-t0:.0f}s)")
    return best_s, best_g

# A: train on subgraph, infer with tr→te edges
ei_tr_to_te = build_graph_tr_to_te(x_tr_np, x_te_np, ei_tr, k=18)
print(f"\nA. Inductive: train on tr subgraph, infer with tr→te edges  (ei_full={ei_tr_to_te.shape[1]})")
run_test("A: tr→te, trained on subgraph", ei_tr_to_te, train_on_tr_only=True)

# B: semi-trans with tr→te edges
print(f"\nB. Semi-trans: tr→te edges, all nodes during training  (ei_full={ei_tr_to_te.shape[1]})")
run_test("B: tr→te, semi-trans", ei_tr_to_te, train_on_tr_only=False)

# C: semi-trans with bidirectional edges  (original v2.py)
ei_bidir = build_graph_bidirectional(x_tr_np, x_te_np, ei_tr, k=18)
print(f"\nC. Semi-trans: bidirectional edges  (ei_full={ei_bidir.shape[1]})")
run_test("C: bidir, semi-trans", ei_bidir, train_on_tr_only=False)

# D: fully transductive on all-node kNN
ei_trans = build_graph_transductive(x_tr_np, x_te_np, k=18)
print(f"\nD. Fully transductive: kNN on ALL nodes  (ei_trans={ei_trans.shape[1]})")
run_test("D: full transductive", ei_trans, train_on_tr_only=False)

# E: original graph from GNN_dual_v2 (the ~96% result from probe_composite.py)
ei_orig = g2['edge_index'].long()
# Reorder edges to match reordered node indices
node_map = {old: new for new, old in enumerate(reorder)}
src_orig = torch.tensor([node_map[s.item()] for s in ei_orig[0]], dtype=torch.long)
dst_orig = torch.tensor([node_map[d.item()] for d in ei_orig[1]], dtype=torch.long)
ei_orig_ro = torch.stack([src_orig, dst_orig])
print(f"\nE. Original graph (GNN_dual_v2 reordered): {ei_orig_ro.shape[1]} edges")
run_test("E: original graph, semi-trans", ei_orig_ro, train_on_tr_only=False)
