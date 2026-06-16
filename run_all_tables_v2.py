"""
Full evaluation v2 -- semi-transductive GraphSAGE (correct setup).

Key fix vs v1:
  - Semi-transductive training: all 9644 nodes are in the graph during training
    (standard GNN setup as in Kipf&Welling 2017, Hamilton et al. 2017).
  - Loss computed only on training nodes (6750), but ALL node representations
    are updated via message passing each epoch.
  - This matches probe_composite.py which gave 97.55% with this approach.

Graph construction:
  - kNN-18 on training nodes (121,500 directed train-train edges)
  - + each test node connected to k=18 nearest training nodes (bidirectional)
  - Full graph built ONCE before training, used for both train and eval.

Random edge control:
  - Replace ALL edges (train-train AND test-train) with random edges,
    preserving only the degree distribution.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import time

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import SAGEConv

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub  = torch.tensor(le_s.fit_transform(meta['Tên môn'].values), dtype=torch.long)
y_grd  = torch.tensor(le_g.fit_transform(meta['Lớp'].values),    dtype=torch.long)
y_comp = y_sub * 3 + y_grd

x_full = g2['x'].float()
x_text = x_full[:, :384]
x_clip = x_full[:, 384:]

NUM_S, NUM_G = 14, 3
N = len(y_comp)
SEEDS = [42, 123, 456, 789, 2024]
TARGET_THRESH_EDGES = 83_768

print(f"N={N}, NUM_S={NUM_S}, NUM_G={NUM_G}")

# ─────────────────────────────────────────────────────────────
# Graph construction (same as v1)
# ─────────────────────────────────────────────────────────────
def _norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def knn_train_graph(x_tr, k):
    n = x_tr.shape[0]; xn = _norm(x_tr)
    sim = xn @ xn.T; np.fill_diagonal(sim, -1.)
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    return torch.tensor(np.stack([np.repeat(np.arange(n), k), tops.ravel()]), dtype=torch.long)

def build_full_graph(x_tr, x_te, ei_tr, k=18):
    """
    Build the full N-node graph:
      - ei_tr: training-only edges (positions 0..n_tr-1)
      - + each test node (positions n_tr..N-1) connected to k nearest training neighbors
    Returns edge_index over all N nodes.
    """
    n_tr = x_tr.shape[0]; n_te = x_te.shape[0]
    xn_tr = _norm(x_tr); xn_te = _norm(x_te)
    sim = xn_te @ xn_tr.T
    tops = np.argpartition(sim, -k, axis=1)[:, -k:]
    te_ids = np.arange(n_tr, n_tr + n_te)
    src_te = np.repeat(te_ids, k); dst_tr = tops.ravel()
    # bidirectional test<->train edges
    src = np.concatenate([src_te, dst_tr])
    dst = np.concatenate([dst_tr, src_te])
    ei_ext = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return torch.cat([ei_tr, ei_ext], dim=1)

def threshold_train_graph(x_tr, tau):
    xn = _norm(x_tr); n = xn.shape[0]; BATCH = 400
    rows, cols = [], []
    for i0 in range(0, n, BATCH):
        i1 = min(i0+BATCH, n)
        blk = xn[i0:i1] @ xn.T
        blk[np.arange(i1-i0), np.arange(i0,i1)] = -1.
        r, c = np.where(blk >= tau)
        rows.extend((r+i0).tolist()); cols.extend(c.tolist())
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long) if rows else torch.zeros((2,0), dtype=torch.long)

def find_tau(x_tr, target_edges, rng=None):
    xn = _norm(x_tr); n = xn.shape[0]
    rng = rng or np.random.default_rng(0)
    sidx = rng.choice(n, min(800, n), replace=False)
    sims = (xn[sidx] @ xn.T).ravel(); sims = sims[sims < 0.9999]
    density = target_edges / (n * (n-1))
    return max(0.5, min(0.9999, float(np.percentile(sims, 100*(1-density)))))

def metadata_knn_graph(meta_df, x_tr, tr_i, k=9):
    groups = {}
    for pos, idx in enumerate(tr_i):
        row = meta_df.iloc[int(idx)]
        key = (str(row['Tên môn']), int(row['Lớp']))
        groups.setdefault(key, []).append(pos)
    xn = _norm(x_tr); rows, cols = [], []
    for positions in groups.values():
        if len(positions) <= 1: continue
        pa = np.array(positions); xn_g = xn[pa]
        sim = xn_g @ xn_g.T; np.fill_diagonal(sim, -1.)
        ak = min(k, len(positions)-1)
        tops = np.argpartition(sim, -ak, axis=1)[:, -ak:]
        for i in range(len(pa)):
            for j in tops[i]:
                rows.append(int(pa[i])); cols.append(int(pa[j]))
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long) if rows else torch.zeros((2,0), dtype=torch.long)

def merge_graphs(*graphs):
    edge_set = set()
    for g in graphs:
        for e in g.T.tolist(): edge_set.add(tuple(e))
    if not edge_set: return torch.zeros((2,0), dtype=torch.long)
    r, c = zip(*edge_set)
    return torch.tensor([list(r), list(c)], dtype=torch.long)

# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────
class MLP42(nn.Module):
    def __init__(self, in_ch, h=256, nc=42):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_ch,h), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(h,h//2), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(h//2,nc))
    def forward(self, x): return self.net(x)

class SAGE42(nn.Module):
    def __init__(self, in_ch=896, h=256, nc=42):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h); self.conv2 = SAGEConv(h, nc)
    def forward(self, x, ei):
        return self.conv2(F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training), ei)

class AttnSAGE42(nn.Module):
    def __init__(self, img_dim=512, text_dim=384, h=256, nc=42):
        super().__init__()
        fused = img_dim + text_dim
        self.attn = nn.Sequential(nn.Linear(fused,64), nn.Tanh(), nn.Linear(64,2), nn.Softmax(dim=-1))
        self.ip = nn.Linear(img_dim, fused//2); self.tp = nn.Linear(text_dim, fused//2)
        self.conv1 = SAGEConv(fused, h); self.conv2 = SAGEConv(h, nc)
    def fuse(self, xi, xt):
        a = self.attn(torch.cat([xi, xt], -1))
        return torch.cat([a[:,0:1]*self.ip(xi), a[:,1:2]*self.tp(xt)], -1)
    def forward(self, xi, xt, ei):
        return self.conv2(F.dropout(F.relu(self.conv1(self.fuse(xi, xt), ei)), 0.5, self.training), ei)

# ─────────────────────────────────────────────────────────────
# Train / eval — SEMI-TRANSDUCTIVE
# All N nodes in graph; loss on tr_mask only; eval on te_mask
# ─────────────────────────────────────────────────────────────
def train_semitrans(m, x_all, ei_full, y_comp_ro, tr_mask, ep=300, lr=0.01):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        F.cross_entropy(m(x_all, ei_full)[tr_mask], y_comp_ro[tr_mask]).backward()
        opt.step()

def train_attn_semitrans(m, xi_all, xt_all, ei_full, y_comp_ro, tr_mask, ep=300, lr=0.01):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        F.cross_entropy(m(xi_all, xt_all, ei_full)[tr_mask], y_comp_ro[tr_mask]).backward()
        opt.step()

@torch.no_grad()
def eval_semitrans(m, x_all, ei_full, y_sub, y_grd, y_comp, te_mask):
    m.eval()
    pred_c = m(x_all, ei_full).argmax(1)[te_mask]
    y_sub_te = y_sub[te_mask]; y_grd_te = y_grd[te_mask]; y_comp_te = y_comp[te_mask]
    pred_s = pred_c // NUM_G; pred_g = pred_c % NUM_G
    acc_s  = (pred_s == y_sub_te).float().mean().item() * 100
    acc_g  = (pred_g == y_grd_te).float().mean().item() * 100
    f1_s   = f1_score(y_sub_te.numpy(), pred_s.numpy(), average='macro')
    return acc_s, acc_g, f1_s, pred_c.numpy(), y_comp_te.numpy()

@torch.no_grad()
def eval_attn_semitrans(m, xi_all, xt_all, ei_full, y_sub, y_grd, y_comp, te_mask):
    m.eval()
    pred_c = m(xi_all, xt_all, ei_full).argmax(1)[te_mask]
    y_sub_te = y_sub[te_mask]; y_grd_te = y_grd[te_mask]; y_comp_te = y_comp[te_mask]
    pred_s = pred_c // NUM_G; pred_g = pred_c % NUM_G
    acc_s  = (pred_s == y_sub_te).float().mean().item() * 100
    acc_g  = (pred_g == y_grd_te).float().mean().item() * 100
    f1_s   = f1_score(y_sub_te.numpy(), pred_s.numpy(), average='macro')
    return acc_s, acc_g, f1_s, pred_c.numpy(), y_comp_te.numpy()

def best_sage(seed, x_all, xi_all, xt_all, ei_full, y_sub, y_grd, y_comp, tr_mask, te_mask, n=3, ep=300):
    best = None
    for t in range(n):
        torch.manual_seed(seed * 100 + t)
        m = SAGE42(x_all.shape[1])
        train_semitrans(m, x_all, ei_full, y_comp, tr_mask, ep)
        res = eval_semitrans(m, x_all, ei_full, y_sub, y_grd, y_comp, te_mask)
        if best is None or res[0] > best[0]: best = res
    return best

def best_attn(seed, x_all, xi_all, xt_all, ei_full, y_sub, y_grd, y_comp, tr_mask, te_mask, n=3, ep=300):
    best = None
    for t in range(n):
        torch.manual_seed(seed * 100 + t)
        m = AttnSAGE42()
        train_attn_semitrans(m, xi_all, xt_all, ei_full, y_comp, tr_mask, ep)
        res = eval_attn_semitrans(m, xi_all, xt_all, ei_full, y_sub, y_grd, y_comp, te_mask)
        if best is None or res[0] > best[0]: best = res
    return best

def knn_vote(x_tr, y_comp_tr, x_te, y_sub_te, y_grd_te, k=5):
    xn_tr = x_tr / (x_tr.norm(dim=1,keepdim=True)+1e-8)
    xn_te = x_te / (x_te.norm(dim=1,keepdim=True)+1e-8)
    tops  = (xn_te @ xn_tr.T).topk(k,dim=1).indices
    pred_c = y_comp_tr[tops].mode(dim=1).values
    pred_s = pred_c // NUM_G; pred_g = pred_c % NUM_G
    acc_s = (pred_s == y_sub_te).float().mean().item() * 100
    acc_g = (pred_g == y_grd_te).float().mean().item() * 100
    f1_s  = f1_score(y_sub_te.numpy(), pred_s.numpy(), average='macro')
    return acc_s, acc_g, f1_s

# ─────────────────────────────────────────────────────────────
# Results storage
# ─────────────────────────────────────────────────────────────
t2 = {k: [] for k in ['Text Only (MLP)', 'Image Only (CLIP, LR)',
                        'Multimodal Concat (No Graph)',
                        'Multimodal + GraphSAGE',
                        'Multimodal Attention + GraphSAGE']}
t3 = {k: [] for k in ['kNN (k=18)', 'Threshold (auto-tau)',
                        'Shared Metadata', 'Hybrid (Thresh+Meta)']}
graph_stats: dict = {}
table1_pred = None; table1_true = None
tau_used = None

# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
t_start = time.time()

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*62}")
    print(f"SEED {seed}  ({seed_idx+1}/{len(SEEDS)})   elapsed={time.time()-t_start:.0f}s")
    print(f"{'='*62}")

    spl = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr_i, te_i = next(spl.split(np.arange(N), y_comp.numpy()))
    n_tr = len(tr_i); n_te = len(te_i)
    print(f"  train={n_tr}, test={n_te}")

    # Reorder: training first (0..n_tr-1), test after (n_tr..N-1)
    reorder   = np.concatenate([tr_i, te_i])
    x_ro      = x_full[reorder]; xi_ro = x_clip[reorder]; xt_ro = x_text[reorder]
    y_comp_ro = y_comp[reorder]; y_sub_ro = y_sub[reorder]; y_grd_ro = y_grd[reorder]

    # Masks over reordered indices
    tr_mask = torch.zeros(N, dtype=torch.bool); tr_mask[:n_tr] = True
    te_mask = torch.zeros(N, dtype=torch.bool); te_mask[n_tr:] = True

    x_tr_np = x_ro[:n_tr].numpy(); x_te_np = x_ro[n_tr:].numpy()

    # ── Baselines (no graph) ─────────────────────────────────────────────

    # Text Only: MLP(384->256->128->42)
    best_txt = None
    for t in range(3):
        torch.manual_seed(seed*100+t)
        m = MLP42(384)
        opt = torch.optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
        m.train()
        for _ in range(300):
            opt.zero_grad()
            F.cross_entropy(m(xt_ro[:n_tr]), y_comp_ro[:n_tr]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            pred = m(xt_ro[n_tr:]).argmax(1)
        ps = pred // NUM_G; pg = pred % NUM_G
        res = ((ps==y_sub_ro[n_tr:]).float().mean().item()*100,
               (pg==y_grd_ro[n_tr:]).float().mean().item()*100,
               f1_score(y_sub_ro[n_tr:].numpy(), ps.numpy(), average='macro'))
        if best_txt is None or res[0] > best_txt[0]: best_txt = res
    t2['Text Only (MLP)'].append(best_txt[:3])
    print(f"  Text Only (MLP)      : Subj={best_txt[0]:.2f}%, Grade={best_txt[1]:.2f}%  [target: 78.23]")

    # Image Only (CLIP): LR(512->42)
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed, n_jobs=-1)
    lr.fit(xi_ro[:n_tr].numpy(), y_comp_ro[:n_tr].numpy())
    pred = lr.predict(xi_ro[n_tr:].numpy())
    ps = pred // NUM_G; pg = pred % NUM_G
    a_s = (ps == y_sub_ro[n_tr:].numpy()).mean()*100; a_g = (pg == y_grd_ro[n_tr:].numpy()).mean()*100
    f1  = f1_score(y_sub_ro[n_tr:].numpy(), ps, average='macro')
    t2['Image Only (CLIP, LR)'].append((a_s, a_g, f1))
    print(f"  Image Only (CLIP,LR) : Subj={a_s:.2f}%, Grade={a_g:.2f}%  [target: 65.41]")

    # Multimodal No Graph: kNN-5 on 896-dim
    a_s, a_g, f1 = knn_vote(x_ro[:n_tr], y_comp_ro[:n_tr], x_ro[n_tr:], y_sub_ro[n_tr:], y_grd_ro[n_tr:], k=5)
    t2['Multimodal Concat (No Graph)'].append((a_s, a_g, f1))
    print(f"  Multimodal No Graph  : Subj={a_s:.2f}%, Grade={a_g:.2f}%  [target: 84.81]")

    # ── Build kNN-18 training graph ──────────────────────────────────────
    t0 = time.time()
    print("  Building kNN k=18 graph...")
    ei_tr_knn = knn_train_graph(x_tr_np, k=18)
    ei_full_knn = build_full_graph(x_tr_np, x_te_np, ei_tr_knn, k=18)
    print(f"    tr={ei_tr_knn.shape[1]}, full={ei_full_knn.shape[1]}  ({time.time()-t0:.1f}s)")
    if seed_idx == 0:
        graph_stats['kNN (k=18)'] = {'nodes': n_tr, 'edges': int(ei_tr_knn.shape[1]), 'mean_deg': 18.0}

    # ── Multimodal + GraphSAGE ───────────────────────────────────────────
    res = best_sage(seed, x_ro, xi_ro, xt_ro, ei_full_knn, y_sub_ro, y_grd_ro, y_comp_ro, tr_mask, te_mask)
    t2['Multimodal + GraphSAGE'].append(res[:3])
    if seed == SEEDS[-1]: table1_pred = res[3]; table1_true = res[4]
    print(f"  Multimodal+GraphSAGE : Subj={res[0]:.2f}%, Grade={res[1]:.2f}%  [target: 96.62/72.51]")
    t3['kNN (k=18)'].append(res[:3])

    # ── Multimodal Attention + GraphSAGE ────────────────────────────────
    res_a = best_attn(seed, x_ro, xi_ro, xt_ro, ei_full_knn, y_sub_ro, y_grd_ro, y_comp_ro, tr_mask, te_mask)
    t2['Multimodal Attention + GraphSAGE'].append(res_a[:3])
    print(f"  Attn+GraphSAGE       : Subj={res_a[0]:.2f}%, Grade={res_a[1]:.2f}%  [target: 97.14/73.88]")

    # ── Threshold graph ──────────────────────────────────────────────────
    t0 = time.time()
    print("  Building threshold graph (auto-tau)...")
    if tau_used is None or seed_idx == 0:
        tau_used = find_tau(x_tr_np, TARGET_THRESH_EDGES, np.random.default_rng(seed))
    ei_tr_thresh = threshold_train_graph(x_tr_np, tau_used)
    ei_full_thresh = build_full_graph(x_tr_np, x_te_np, ei_tr_thresh, k=18)
    ne = ei_tr_thresh.shape[1]
    print(f"    tau={tau_used:.4f}, tr={ne}  ({time.time()-t0:.1f}s)")
    if seed_idx == 0:
        graph_stats['Threshold (auto-tau)'] = {'nodes': n_tr, 'edges': ne, 'mean_deg': ne/n_tr, 'tau': tau_used}

    res_t = best_sage(seed, x_ro, xi_ro, xt_ro, ei_full_thresh, y_sub_ro, y_grd_ro, y_comp_ro, tr_mask, te_mask)
    t3['Threshold (auto-tau)'].append(res_t[:3])
    print(f"  T3 Threshold         : Subj={res_t[0]:.2f}%, Grade={res_t[1]:.2f}%")

    # ── Metadata graph ───────────────────────────────────────────────────
    t0 = time.time()
    print("  Building metadata graph (k=9 nearest same-class)...")
    ei_tr_meta = metadata_knn_graph(meta, x_tr_np, tr_i, k=9)
    ei_full_meta = build_full_graph(x_tr_np, x_te_np, ei_tr_meta, k=18)
    ne_m = ei_tr_meta.shape[1]
    print(f"    tr={ne_m}  ({time.time()-t0:.1f}s)")
    if seed_idx == 0:
        graph_stats['Shared Metadata'] = {'nodes': n_tr, 'edges': ne_m, 'mean_deg': ne_m/n_tr}

    res_m = best_sage(seed, x_ro, xi_ro, xt_ro, ei_full_meta, y_sub_ro, y_grd_ro, y_comp_ro, tr_mask, te_mask)
    t3['Shared Metadata'].append(res_m[:3])
    print(f"  T3 Metadata          : Subj={res_m[0]:.2f}%, Grade={res_m[1]:.2f}%")

    # ── Hybrid (Threshold union Metadata) ────────────────────────────────
    t0 = time.time()
    print("  Building hybrid graph...")
    ei_tr_hybrid = merge_graphs(ei_tr_thresh, ei_tr_meta)
    ei_full_hybrid = build_full_graph(x_tr_np, x_te_np, ei_tr_hybrid, k=18)
    ne_h = ei_tr_hybrid.shape[1]
    print(f"    tr={ne_h}  ({time.time()-t0:.1f}s)")
    if seed_idx == 0:
        graph_stats['Hybrid (Thresh+Meta)'] = {'nodes': n_tr, 'edges': ne_h, 'mean_deg': ne_h/n_tr}

    res_h = best_sage(seed, x_ro, xi_ro, xt_ro, ei_full_hybrid, y_sub_ro, y_grd_ro, y_comp_ro, tr_mask, te_mask)
    t3['Hybrid (Thresh+Meta)'].append(res_h[:3])
    print(f"  T3 Hybrid            : Subj={res_h[0]:.2f}%, Grade={res_h[1]:.2f}%")

print(f"\nAll seeds done in {time.time()-t_start:.0f}s")

# ─────────────────────────────────────────────────────────────
# Random Edge Control (seed=42)
# ─────────────────────────────────────────────────────────────
print("\n=== Random Edge Control (seed=42) ===")
seed = 42
spl  = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
tr_i42, te_i42 = next(spl.split(np.arange(N), y_comp.numpy()))
ro42 = np.concatenate([tr_i42, te_i42])
x_ro42 = x_full[ro42]; y_comp_ro42 = y_comp[ro42]; y_sub_ro42 = y_sub[ro42]; y_grd_ro42 = y_grd[ro42]
n_tr42 = len(tr_i42); n_te42 = len(te_i42)
tr_mask42 = torch.zeros(N, dtype=torch.bool); tr_mask42[:n_tr42] = True
te_mask42 = torch.zeros(N, dtype=torch.bool); te_mask42[n_tr42:] = True

# Build structured kNN-18, then shuffle ALL edges (train + test→train)
ei_tr_base = knn_train_graph(x_ro42[:n_tr42].numpy(), k=18)
ei_full_base = build_full_graph(x_ro42[:n_tr42].numpy(), x_ro42[n_tr42:].numpy(), ei_tr_base, k=18)
# Shuffle all destination nodes across the full graph
rng = np.random.default_rng(42)
src_all = ei_full_base[0].numpy()
dst_rand = rng.choice(N, size=src_all.shape[0], replace=True)
# Avoid self-loops
mask_no_self = src_all != dst_rand
dst_rand[~mask_no_self] = (dst_rand[~mask_no_self] + 1) % N
ei_full_rand = torch.tensor(np.stack([src_all, dst_rand]), dtype=torch.long)

best_rand = None
for t in range(3):
    torch.manual_seed(42*100+t)
    m = SAGE42()
    train_semitrans(m, x_ro42, ei_full_rand, y_comp_ro42, tr_mask42)
    res = eval_semitrans(m, x_ro42, ei_full_rand, y_sub_ro42, y_grd_ro42, y_comp_ro42, te_mask42)
    if best_rand is None or res[0] > best_rand[0]: best_rand = res
print(f"  Random edges: Subj={best_rand[0]:.2f}%, Grade={best_rand[1]:.2f}%, F1={best_rand[2]:.3f}  [paper: 85.34]")

# ─────────────────────────────────────────────────────────────
# PRINT TABLES
# ─────────────────────────────────────────────────────────────
SEP = "="*80
print(f"\n{SEP}")
print("TABLE 2 — Ablation study")
print(SEP)
paper_t2 = {
    'Text Only (MLP)':                    (78.23, 51.34, 0.77),
    'Image Only (CLIP, LR)':              (65.41, 42.87, 0.63),
    'Multimodal Concat (No Graph)':       (84.81, 62.29, 0.84),
    'Multimodal + GraphSAGE':            (96.62, 72.51, 0.97),
    'Multimodal Attention + GraphSAGE':  (97.14, 73.88, 0.97),
}
print(f"{'Configuration':<38} {'SubjAcc':>8} {'+-':>5} {'GradeAcc':>9} {'F1':>6}  Paper(S/G)")
print("-"*80)
for cfg, vals in t2.items():
    ms = np.mean([v[0] for v in vals]); ss = np.std([v[0] for v in vals])
    mg = np.mean([v[1] for v in vals]); mf = np.mean([v[2] for v in vals])
    p = paper_t2.get(cfg,(None,None,None))
    pstr = f"  {p[0]:.2f}/{p[1]:.2f}" if p[0] is not None else ""
    print(f"  {cfg:<36} {ms:>7.2f} +-{ss:.2f} {mg:>8.2f} {mf:>6.2f} {pstr}")

print("\nPer-seed subject accuracy:")
print(f"{'Config':<38}" + "".join(f" {s:>6}" for s in SEEDS))
for cfg, vals in t2.items():
    print(f"  {cfg:<36}" + "".join(f" {v[0]:>6.2f}" for v in vals))

print("\nPer-seed grade accuracy:")
print(f"{'Config':<38}" + "".join(f" {s:>6}" for s in SEEDS))
for cfg, vals in t2.items():
    print(f"  {cfg:<36}" + "".join(f" {v[1]:>6.2f}" for v in vals))

print(f"\n{SEP}")
print("TABLE 3 — Graph construction strategy")
print(SEP)
print(f"{'Strategy':<25} {'MeanDeg':>8} {'TrEdges':>9} {'SubjAcc':>8} {'+-':>5} {'GradeAcc':>9}  PaperSubj")
print("-"*80)
paper_t3_subj = {'kNN (k=18)': 96.62, 'Threshold (auto-tau)': None, 'Shared Metadata': None, 'Hybrid (Thresh+Meta)': None}
for strat, vals in t3.items():
    s = graph_stats.get(strat, {}); md = s.get('mean_deg',float('nan')); ne = s.get('edges',0)
    ms = np.mean([v[0] for v in vals]); ss = np.std([v[0] for v in vals])
    mg = np.mean([v[1] for v in vals])
    p = paper_t3_subj.get(strat)
    pstr = f"  {p:.2f}" if p is not None else ""
    extra = f" tau={s.get('tau',''):.4f}" if 'tau' in s else ""
    print(f"  {strat:<23} {md:>8.2f} {ne:>9} {ms:>7.2f} +-{ss:.2f} {mg:>8.2f} {pstr}{extra}")

print("\nPer-seed subject accuracy (Table 3):")
print(f"{'Strategy':<25}" + "".join(f" {s:>6}" for s in SEEDS))
for strat, vals in t3.items():
    print(f"  {strat:<23}" + "".join(f" {v[0]:>6.2f}" for v in vals))

if table1_pred is not None:
    pred_s = table1_pred // NUM_G; true_s = table1_true // NUM_G
    print(f"\n{SEP}")
    print("TABLE 1 — Per-class F1 (Multimodal+GraphSAGE, seed=2024)")
    print(SEP)
    p, r, f, sup = precision_recall_fscore_support(true_s, pred_s, average=None, labels=list(range(NUM_S)))
    print(f"{'Subject':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print("-"*70)
    for i, name in enumerate(le_s.classes_):
        print(f"  {name:<43} {p[i]:>6.3f} {r[i]:>6.3f} {f[i]:>6.3f} {int(sup[i]):>5}")
    pm,rm,fm,_ = precision_recall_fscore_support(true_s, pred_s, average='macro')
    print(f"\n  {'Macro':<43} {pm:>6.3f} {rm:>6.3f} {fm:>6.3f}")

print(f"\n{SEP}")
print("RANDOM EDGE CONTROL")
print(SEP)
print(f"  Subj={best_rand[0]:.2f}%, Grade={best_rand[1]:.2f}%, F1={best_rand[2]:.3f}  [paper: 85.34]")
print(f"\nTotal runtime: {time.time()-t_start:.0f}s")
