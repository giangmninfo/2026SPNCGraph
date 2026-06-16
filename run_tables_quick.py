"""Quick 1-seed test of run_all_tables.py logic."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import SAGEConv

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub  = torch.tensor(le_s.fit_transform(meta['Tên môn'].values), dtype=torch.long)
y_grd  = torch.tensor(le_g.fit_transform(meta['Lớp'].values),    dtype=torch.long)
y_comp = y_sub * 3 + y_grd
x_full = g2['x'].float()
x_text = x_full[:, :384]; x_clip = x_full[:, 384:]
NUM_S, NUM_G = 14, 3; N = len(y_comp)

def _norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def knn_train_graph(x_tr, k):
    n = x_tr.shape[0]; xn = _norm(x_tr)
    sim = xn @ xn.T; np.fill_diagonal(sim, -1.)
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    src = np.repeat(np.arange(n), k); dst = tops.ravel()
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)

def threshold_train_graph(x_tr, tau):
    xn = _norm(x_tr); BATCH = 500; n = xn.shape[0]
    rows, cols = [], []
    for i0 in range(0, n, BATCH):
        i1 = min(i0+BATCH, n)
        blk = xn[i0:i1] @ xn.T
        blk[np.arange(i1-i0), np.arange(i0, i1)] = -1.
        r, c = np.where(blk >= tau)
        rows.extend((r+i0).tolist()); cols.extend(c.tolist())
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)

def metadata_knn_train_graph(meta_df, x_tr, tr_i, k=9):
    groups = {}
    for pos, idx in enumerate(tr_i):
        row = meta_df.iloc[int(idx)]
        key = (str(row['Tên môn']), int(row['Lớp']))
        groups.setdefault(key, []).append(pos)
    xn = _norm(x_tr); rows, cols = [], []
    for key, positions in groups.items():
        if len(positions) <= 1: continue
        pa = np.array(positions); xn_g = xn[pa]
        sim = xn_g @ xn_g.T; np.fill_diagonal(sim, -1.)
        ak = min(k, len(positions)-1)
        tops = np.argpartition(sim, -ak, axis=1)[:, -ak:]
        for i in range(len(pa)):
            for j in tops[i]:
                rows.append(int(pa[i])); cols.append(int(pa[j]))
    if not rows: return torch.zeros((2,0), dtype=torch.long)
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)

def extend_to_test(ei_tr, x_tr, x_te, k=18):
    n_tr = x_tr.shape[0]
    xn_tr = _norm(x_tr); xn_te = _norm(x_te)
    sim = xn_te @ xn_tr.T
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    te_ids = np.arange(n_tr, n_tr + x_te.shape[0])
    src_te = np.repeat(te_ids, k); dst_tr = tops.ravel()
    src = np.concatenate([src_te, dst_tr]); dst = np.concatenate([dst_tr, src_te])
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
n_tr = len(tr_i); n_te = len(te_i)
print(f"train={n_tr}, test={n_te}")
reorder = np.concatenate([tr_i, te_i])
x_ro = x_full[reorder]; y_comp_ro = y_comp[reorder]
y_sub_ro = y_sub[reorder]; y_grd_ro = y_grd[reorder]
x_tr_np = x_ro[:n_tr].numpy(); x_te_np = x_ro[n_tr:].numpy()
y_comp_tr = y_comp_ro[:n_tr]
y_sub_te = y_sub_ro[n_tr:]; y_grd_te = y_grd_ro[n_tr:]; y_comp_te = y_comp_ro[n_tr:]

# ── LR baselines
lr = LogisticRegression(max_iter=500, random_state=seed, n_jobs=-1)
lr.fit(x_ro[:n_tr, :384].numpy(), y_comp_tr.numpy())
pred = lr.predict(x_ro[n_tr:, :384].numpy())
ps = pred // NUM_G
a_s = (ps == y_sub_te.numpy()).mean() * 100
print(f"Text Only LR: Subj={a_s:.2f}%  (target: 78.23%)")

lr2 = LogisticRegression(max_iter=500, random_state=seed, n_jobs=-1)
lr2.fit(x_ro[:n_tr, 384:].numpy(), y_comp_tr.numpy())
pred2 = lr2.predict(x_ro[n_tr:, 384:].numpy())
ps2 = pred2 // NUM_G
a_s2 = (ps2 == y_sub_te.numpy()).mean() * 100
print(f"Image Only LR: Subj={a_s2:.2f}%  (target: 65.41%)")

# ── kNN No Graph
x_tr_t = x_ro[:n_tr]; x_te_t = x_ro[n_tr:]
xn_tr = x_tr_t / (x_tr_t.norm(dim=1, keepdim=True)+1e-8)
xn_te = x_te_t / (x_te_t.norm(dim=1, keepdim=True)+1e-8)
tops  = (xn_te @ xn_tr.T).topk(5, dim=1).indices
pred_c = y_comp_tr[tops].mode(dim=1).values
ps_k = pred_c // NUM_G
a_s3 = (ps_k == y_sub_te).float().mean().item() * 100
print(f"kNN No Graph: Subj={a_s3:.2f}%  (target: 84.81%)")

# ── kNN k=18 graph + SAGE42
print("Building kNN-18 graph...")
ei_tr = knn_train_graph(x_tr_np, k=18)
ei_ext = extend_to_test(ei_tr, x_tr_np, x_te_np, k=18)
print(f"  ei_tr: {ei_tr.shape}, ei_ext: {ei_ext.shape}")

torch.manual_seed(42)
m = SAGE42()
opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
m.train()
for ep in range(100):  # Quick test: 100 epochs
    opt.zero_grad()
    F.cross_entropy(m(x_ro[:n_tr], ei_tr), y_comp_tr).backward()
    opt.step()
m.eval()
with torch.no_grad():
    pred_c2 = m(x_ro, ei_ext).argmax(1)[n_tr:]
ps2 = pred_c2 // NUM_G; pg2 = pred_c2 % NUM_G
a_s4 = (ps2 == y_sub_te).float().mean().item() * 100
a_g4 = (pg2 == y_grd_te).float().mean().item() * 100
print(f"SAGE42 100ep: Subj={a_s4:.2f}%, Grade={a_g4:.2f}%  (target: 96.62/72.51)")

# ── Threshold graph
print("Building threshold graph (tau=0.75)...")
ei_tr_thresh = threshold_train_graph(x_tr_np, tau=0.75)
print(f"  Threshold edges: {ei_tr_thresh.shape[1]}  (target: ~83,768)")

# ── Metadata graph
print("Building metadata graph (k=9)...")
ei_tr_meta = metadata_knn_train_graph(meta, x_tr_np, tr_i, k=9)
print(f"  Metadata edges: {ei_tr_meta.shape[1]}  (target: ~59,130)")

# ── Hybrid
edge_set = set(map(tuple, ei_tr.T.tolist()))
for e in ei_tr_meta.T.tolist(): edge_set.add(tuple(e))
ei_tr_hybrid = torch.tensor([list(r) for r in zip(*edge_set)], dtype=torch.long)
print(f"  Hybrid edges: {ei_tr_hybrid.shape[1]}  (target: ~98,010)")
print("Quick test PASSED")
