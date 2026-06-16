"""
run_khungB.py — Comprehensive experiment runner (Khung B spec).

Reads: split_indices.json (frozen indices), graph_data.pt (features)
Writes: results_khungB.json

Architecture decisions (Option B for VisualBERT):
  - VisualBERT replaced by "Fusion MLP" (MLP on 896-dim concat features)
  - All GNN models: dual-head (subject 14-class + grade 3-class)
  - Hidden=128, dropout=0.3, lr=1e-3, wd=0, patience=20, max_ep=200
  - Graph: intra-subject kNN (train) + cosine attachment (val/test)
"""
import sys, io, json, time, random, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import os; os.environ['PYTHONHASHSEED'] = '42'

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from scipy import stats
from scipy.optimize import minimize_scalar
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

# ─── Constants ────────────────────────────────────────────────────────────────
BASE   = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
SPLIT  = r'd:\SPNC_gnnclassifier-main\split_indices.json'
OUT    = r'd:\SPNC_gnnclassifier-main\results_khungB.json'
SEEDS  = [42, 123, 456, 789, 2024]
LR, WD, H, DR, PAT, MAX_EP = 1e-3, 0.0, 128, 0.3, 20, 200
NUM_S, NUM_G = 14, 3
K_SWEEP_VALS = [6, 8, 12, 18, 24]
TAU_SWEEP_VALS = [0.75, 0.79, 0.83, 0.87]

results = {}  # populated throughout; saved as JSON at end

# ─── A: Seed ────────────────────────────────────────────────────────────────
def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    random.seed(s); torch.cuda.manual_seed_all(s)

# ─── B: Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')
le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub  = torch.tensor(le_s.fit_transform(meta['Tên môn'].values), dtype=torch.long)
y_grd  = torch.tensor(le_g.fit_transform(meta['Lớp'].values),    dtype=torch.long)
x_full = g2['x'].float(); N = len(y_sub)
x_text = x_full[:, :384]; x_clip = x_full[:, 384:]

with open(SPLIT, encoding='utf-8') as f:
    sp = json.load(f)
tr_i  = np.array(sp['train']); val_i = np.array(sp['val']); te_i = np.array(sp['test'])
n_tr, n_val, n_te = len(tr_i), len(val_i), len(te_i)
print(f"Split: train={n_tr}, val={n_val}, test={n_te}")

results['split'] = {
    'train': n_tr, 'val': n_val, 'test': n_te,
    'indices_file': 'split_indices.json'
}

# Reordered: train | val | test
reorder    = np.concatenate([tr_i, val_i, te_i])
x_ro       = x_full[reorder]                       # (N, 896) raw
xi_ro      = x_clip[reorder]                       # (N, 512) CLIP
xt_ro      = x_text[reorder]                       # (N, 384) text
y_sub_ro   = y_sub[reorder]
y_grd_ro   = y_grd[reorder]

# L2-normalize fused features for cosine similarity (graph construction)
x_np = x_ro.numpy()
x_norm = (x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

tr_mask  = torch.zeros(N, dtype=torch.bool); tr_mask[:n_tr] = True
val_mask = torch.zeros(N, dtype=torch.bool); val_mask[n_tr:n_tr+n_val] = True
te_mask  = torch.zeros(N, dtype=torch.bool); te_mask[n_tr+n_val:] = True

y_sub_np = y_sub_ro.numpy(); y_grd_np = y_grd_ro.numpy()
x_tr_norm = x_norm[:n_tr]; x_val_norm = x_norm[n_tr:n_tr+n_val]
x_te_norm  = x_norm[n_tr+n_val:]

# ─── C: Model Classes ─────────────────────────────────────────────────────────

class SAGEDual(nn.Module):
    """SAGE Mean aggregator, dual head (subject + grade)."""
    def __init__(self, in_ch=896, h=H, ns=NUM_S, ng=NUM_G, aggr='mean'):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h, aggr=aggr)
        self.conv2 = SAGEConv(h, h, aggr=aggr)
        self.head_s = nn.Linear(h, ns); self.head_g = nn.Linear(h, ng)
    def forward(self, x, ei):
        h = F.dropout(F.relu(self.conv1(x, ei)), DR, self.training)
        h = F.relu(self.conv2(h, ei))
        return self.head_s(h), self.head_g(h)
    def embed(self, x, ei):
        h = F.relu(self.conv1(x, ei))
        return F.relu(self.conv2(h, ei))

class LSTMSAGEConv(nn.Module):
    """SAGEConv with LSTM aggregation (batched padded implementation)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, out_ch, batch_first=True)
        self.lin_root = nn.Linear(in_ch, out_ch, bias=False)
        self.out_ch = out_ch
    def forward(self, x, edge_index):
        n, dev = x.size(0), x.device
        agg = torch.zeros(n, self.out_ch, device=dev)
        src, dst = edge_index
        if src.numel() > 0:
            order = dst.argsort(); src_s = src[order]; dst_s = dst[order]
            ud, cnts = torch.unique_consecutive(dst_s, return_counts=True)
            max_k = cnts.max().item()
            padded  = torch.zeros(len(ud), max_k, x.size(1), device=dev)
            lengths = torch.zeros(len(ud), dtype=torch.long)
            start = 0
            for i, (v, c) in enumerate(zip(ud.tolist(), cnts.tolist())):
                padded[i, :c] = x[src_s[start:start+c]]
                lengths[i] = c; start += c
            packed = nn.utils.rnn.pack_padded_sequence(
                padded, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
            for i, v in enumerate(ud.tolist()): agg[v] = h.squeeze(0)[i]
        return F.relu(agg + self.lin_root(x))

class SAGEDualLSTM(nn.Module):
    def __init__(self, in_ch=896, h=H, ns=NUM_S, ng=NUM_G):
        super().__init__()
        self.conv1 = LSTMSAGEConv(in_ch, h)
        self.conv2 = SAGEConv(h, h, aggr='mean')
        self.head_s = nn.Linear(h, ns); self.head_g = nn.Linear(h, ng)
    def forward(self, x, ei):
        h = F.dropout(self.conv1(x, ei), DR, self.training)
        h = F.relu(self.conv2(h, ei))
        return self.head_s(h), self.head_g(h)

class AttnFuse(nn.Module):
    def forward_fuse(self, xi, xt):
        cat = torch.cat([xi, xt], -1)
        a = torch.softmax(self.attn(cat), -1)
        return torch.cat([a[:,0:1]*self.ip(xi), a[:,1:2]*self.tp(xt)], -1)

class AttnSAGEDual(nn.Module):
    """Cross-modal attention fusion + dual-head SAGE."""
    def __init__(self, img_dim=512, txt_dim=384, h=H, ns=NUM_S, ng=NUM_G):
        super().__init__()
        fused = img_dim + txt_dim
        self.attn = nn.Sequential(nn.Linear(fused,64), nn.Tanh(),
                                  nn.Linear(64,2), nn.Softmax(dim=-1))
        self.ip   = nn.Linear(img_dim, img_dim)
        self.tp   = nn.Linear(txt_dim, txt_dim)
        self.conv1 = SAGEConv(fused, h); self.conv2 = SAGEConv(h, h)
        self.head_s = nn.Linear(h, ns); self.head_g = nn.Linear(h, ng)
    def fuse(self, xi, xt):
        a = self.attn(torch.cat([xi, xt], -1))
        return torch.cat([a[:,0:1]*self.ip(xi), a[:,1:2]*self.tp(xt)], -1)
    def forward(self, xi, xt, ei):
        h = F.dropout(F.relu(self.conv1(self.fuse(xi, xt), ei)), DR, self.training)
        h = F.relu(self.conv2(h, ei))
        return self.head_s(h), self.head_g(h)

class GCNDual(nn.Module):
    def __init__(self, in_ch=896, h=H, ns=NUM_S, ng=NUM_G):
        super().__init__()
        self.conv1 = GCNConv(in_ch, h); self.conv2 = GCNConv(h, h)
        self.head_s = nn.Linear(h, ns); self.head_g = nn.Linear(h, ng)
    def forward(self, x, ei):
        h = F.dropout(F.relu(self.conv1(x, ei)), DR, self.training)
        h = F.relu(self.conv2(h, ei))
        return self.head_s(h), self.head_g(h)

class GATDual(nn.Module):
    def __init__(self, in_ch=896, h=H, heads=4, ns=NUM_S, ng=NUM_G):
        super().__init__()
        hh = max(1, h // heads)
        self.conv1 = GATConv(in_ch, hh, heads=heads, concat=True, dropout=DR)
        self.conv2 = GATConv(hh*heads, h, heads=1, concat=True, dropout=DR)
        self.head_s = nn.Linear(h, ns); self.head_g = nn.Linear(h, ng)
    def forward(self, x, ei):
        h = F.dropout(F.elu(self.conv1(x, ei)), DR, self.training)
        h = F.elu(self.conv2(h, ei))
        return self.head_s(h), self.head_g(h)

class MLPDual(nn.Module):
    def __init__(self, in_ch=896, h=H, ns=NUM_S, ng=NUM_G):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_ch, h*2), nn.ReLU(), nn.Dropout(DR),
                                 nn.Linear(h*2, h), nn.ReLU())
        self.head_s = nn.Linear(h, ns); self.head_g = nn.Linear(h, ng)
    def forward(self, x):
        h = self.net(x)
        return self.head_s(h), self.head_g(h)

# ─── D: Graph Builders ───────────────────────────────────────────────────────

def intra_subj_knn(x_norm_tr, y_sub_tr_np, k):
    """Directed intra-subject kNN on training nodes. No self-loop, no symmetrize."""
    all_src, all_dst = [], []
    for s in np.unique(y_sub_tr_np):
        idx = np.where(y_sub_tr_np == s)[0]
        if len(idx) < 2: continue
        sim = x_norm_tr[idx] @ x_norm_tr[idx].T; np.fill_diagonal(sim, -1.)
        ki = min(k, len(idx)-1)
        tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
        all_src.append(idx[np.repeat(np.arange(len(idx)), ki)])
        all_dst.append(idx[tops.ravel()])
    if not all_src: return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(np.stack([np.concatenate(all_src), np.concatenate(all_dst)]), dtype=torch.long)

def cosine_knn(x_norm_q, x_norm_ref, k, offset_dst):
    """Each query node → k nearest ref nodes by cosine. Returns (train→query) edges."""
    sim = x_norm_q @ x_norm_ref.T  # (n_q, n_ref)
    ki = min(k, x_norm_ref.shape[0])
    tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
    n_q = x_norm_q.shape[0]
    dst = np.repeat(np.arange(n_q) + offset_dst, ki)
    src = tops.ravel()   # ref (train) nodes as source
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)

def feature_knn_train(x_norm_tr, k):
    """Cosine kNN on train nodes, no subject filter."""
    n = x_norm_tr.shape[0]; sim = x_norm_tr @ x_norm_tr.T; np.fill_diagonal(sim, -1.)
    ki = min(k, n-1)
    tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
    src = np.repeat(np.arange(n), ki); dst = tops.ravel()
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)

def threshold_train(x_norm_tr, tau):
    """Connect train pairs with cosine > tau (batched)."""
    n = x_norm_tr.shape[0]; BATCH = 500; rows, cols = [], []
    for i0 in range(0, n, BATCH):
        blk = x_norm_tr[i0:i0+BATCH] @ x_norm_tr.T
        blk[np.arange(min(BATCH, n-i0)), np.arange(i0, min(i0+BATCH, n))] = -1.
        r, c = np.where(blk > tau); rows.extend((r+i0).tolist()); cols.extend(c.tolist())
    if not rows: return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)

def graph_stats(ei, y_sub_np, y_grd_np, n_tr):
    """Compute edge stats: total edges, mean train out-degree, same-subject/grade %."""
    if ei.shape[1] == 0: return {'edges': 0, 'mean_deg': 0., 'same_subj_pct': 0., 'same_grade_pct': 0.}
    tr_mask_edge = (ei[0] < n_tr) & (ei[1] < n_tr)
    ei_tr = ei[:, tr_mask_edge]
    same_s = (y_sub_np[ei[0]] == y_sub_np[ei[1]]).mean() * 100 if ei.shape[1] > 0 else 0.
    same_g = (y_grd_np[ei[0]] == y_grd_np[ei[1]]).mean() * 100 if ei.shape[1] > 0 else 0.
    mean_deg = ei_tr.shape[1] / n_tr if n_tr > 0 else 0.
    return {'edges': int(ei.shape[1]), 'mean_deg': float(mean_deg),
            'same_subj_pct': float(same_s), 'same_grade_pct': float(same_g)}

# ─── E: Training / Evaluation ─────────────────────────────────────────────────

def train_dual_sage(model, x, ei_tr, ei_val_full, y_sub, y_grd, tr_mask, val_mask,
                    attn_mode=False, xi=None, xt=None):
    """Full-batch dual-head GNN with early stopping on val loss."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_vl, pat, best_st = float('inf'), 0, None
    for ep in range(MAX_EP):
        model.train(); opt.zero_grad()
        if attn_mode: ls, lg = model(xi, xt, ei_tr)
        else:         ls, lg = model(x, ei_tr)
        loss = F.cross_entropy(ls[tr_mask], y_sub[tr_mask]) + \
               F.cross_entropy(lg[tr_mask], y_grd[tr_mask])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            if attn_mode: vs, vg = model(xi, xt, ei_val_full)
            else:         vs, vg = model(x, ei_val_full)
            vl = (F.cross_entropy(vs[val_mask], y_sub[val_mask]) +
                  F.cross_entropy(vg[val_mask], y_grd[val_mask])).item()
        if vl < best_vl - 1e-4:
            best_vl = vl; pat = 0
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PAT: break
    if best_st: model.load_state_dict(best_st)
    return model

def train_dual_mlp(model, x, y_sub, y_grd, tr_mask, ei_val_full, val_mask):
    """MLP dual-head with early stopping on val loss."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_vl, pat, best_st = float('inf'), 0, None
    for ep in range(MAX_EP):
        model.train(); opt.zero_grad()
        ls, lg = model(x[tr_mask])
        loss = F.cross_entropy(ls, y_sub[tr_mask]) + F.cross_entropy(lg, y_grd[tr_mask])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vs, vg = model(x[val_mask])
            vl = (F.cross_entropy(vs, y_sub[val_mask]) + F.cross_entropy(vg, y_grd[val_mask])).item()
        if vl < best_vl - 1e-4:
            best_vl = vl; pat = 0; best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PAT: break
    if best_st: model.load_state_dict(best_st)
    return model

def eval_dual(model, x, ei, te_mask, y_sub, y_grd, attn_mode=False, xi=None, xt=None):
    model.eval()
    with torch.no_grad():
        if attn_mode: ls, lg = model(xi, xt, ei)
        else:         ls, lg = model(x, ei)
    pred_s = ls[te_mask].argmax(1); pred_g = lg[te_mask].argmax(1)
    acc_s  = (pred_s == y_sub[te_mask]).float().mean().item() * 100
    acc_g  = (pred_g == y_grd[te_mask]).float().mean().item() * 100
    f1_s   = f1_score(y_sub[te_mask].numpy(), pred_s.numpy(), average='macro', zero_division=0)
    pr, rc, f1c, _ = precision_recall_fscore_support(
        y_sub[te_mask].numpy(), pred_s.numpy(), average='macro', zero_division=0)
    return acc_s, acc_g, float(f1_s), float(pr), float(rc), pred_s, ls[te_mask]

def eval_mlp(model, x, mask, y_sub, y_grd):
    model.eval()
    with torch.no_grad(): ls, lg = model(x[mask])
    pred_s = ls.argmax(1); pred_g = lg.argmax(1)
    acc_s = (pred_s == y_sub[mask]).float().mean().item() * 100
    acc_g = (pred_g == y_grd[mask]).float().mean().item() * 100
    f1_s  = f1_score(y_sub[mask].numpy(), pred_s.numpy(), average='macro', zero_division=0)
    return acc_s, acc_g, float(f1_s)

def run5seeds_sage(model_fn, ei_tr, ei_test, model_kwargs=None, attn_mode=False):
    """Run model_fn over 5 seeds, return per-seed (subj_acc, grade_acc, f1, prec, rec)."""
    mk = model_kwargs or {}
    per = []
    for seed in SEEDS:
        set_seed(seed)
        m = model_fn(**mk)
        m = train_dual_sage(m, x_ro, ei_tr, ei_for_val, y_sub_ro, y_grd_ro,
                            tr_mask, val_mask, attn_mode=attn_mode,
                            xi=xi_ro if attn_mode else None,
                            xt=xt_ro if attn_mode else None)
        r = eval_dual(m, x_ro, ei_test, te_mask, y_sub_ro, y_grd_ro,
                      attn_mode=attn_mode,
                      xi=xi_ro if attn_mode else None,
                      xt=xt_ro if attn_mode else None)
        per.append(r[:5])
        print(f"    seed={seed}: subj={r[0]:.2f}%, grade={r[1]:.2f}%, f1={r[2]:.3f}")
    return per

def summarize(per):
    subjs = [r[0] for r in per]; grds = [r[1] for r in per]; f1s = [r[2] for r in per]
    return {
        'per_seed_subj':  [round(s, 4) for s in subjs],
        'per_seed_grade': [round(g, 4) for g in grds],
        'subj_mean': round(float(np.mean(subjs)), 4),
        'subj_sd':   round(float(np.std(subjs)), 4),
        'grade_mean': round(float(np.mean(grds)), 4),
        'f1_macro':   round(float(np.mean(f1s)), 4),
        'prec_macro': round(float(np.mean([r[3] for r in per])), 4),
        'rec_macro':  round(float(np.mean([r[4] for r in per])), 4),
    }

# ─── F: k-sweep (seed=42, val accuracy of SAGE Mean) ─────────────────────────
print("\n=== k-sweep (seed=42, val acc) ===")
k_sweep_results = {}
best_k, best_k_acc = K_SWEEP_VALS[0], -1.0
for k in K_SWEEP_VALS:
    ei_k_tr = intra_subj_knn(x_tr_norm, y_sub_np[:n_tr], k)
    ei_k_att_val = cosine_knn(x_val_norm, x_tr_norm, k, n_tr)
    ei_k_val_full = torch.cat([ei_k_tr, ei_k_att_val], 1)
    ei_k_att_te = cosine_knn(x_te_norm, x_tr_norm, k, n_tr+n_val)
    ei_k_test = torch.cat([ei_k_tr, ei_k_att_val, ei_k_att_te], 1)
    set_seed(42)
    m = SAGEDual()
    m = train_dual_sage(m, x_ro, ei_k_tr, ei_k_val_full, y_sub_ro, y_grd_ro, tr_mask, val_mask)
    m.eval()
    with torch.no_grad(): vs, _ = m(x_ro, ei_k_val_full)
    val_acc = (vs[val_mask].argmax(1) == y_sub_ro[val_mask]).float().mean().item() * 100
    k_sweep_results[str(k)] = round(val_acc, 3)
    print(f"  k={k:2d}: val_acc={val_acc:.2f}%  (ei_tr={ei_k_tr.shape[1]})")
    if val_acc > best_k_acc: best_k_acc = val_acc; best_k = k

print(f"  → k* = {best_k}  (val_acc={best_k_acc:.2f}%)")
results['k_star'] = int(best_k)
results['k_sweep'] = k_sweep_results

# ─── G: Build graphs with k* ─────────────────────────────────────────────────
k = best_k
print(f"\nBuilding graphs with k*={k}...")
ei_tr        = intra_subj_knn(x_tr_norm, y_sub_np[:n_tr], k)
ei_att_val   = cosine_knn(x_val_norm,   x_tr_norm, k, n_tr)
ei_att_te    = cosine_knn(x_te_norm,    x_tr_norm, k, n_tr+n_val)
ei_for_val   = torch.cat([ei_tr, ei_att_val], 1)            # for val monitoring
ei_for_test  = torch.cat([ei_tr, ei_att_val, ei_att_te], 1) # for test eval
print(f"  Intra-subj train: {ei_tr.shape[1]} edges, "
      f"val att: {ei_att_val.shape[1]}, test att: {ei_att_te.shape[1]}")

# Same-grade % in training graph
sg = graph_stats(ei_tr, y_sub_np, y_grd_np, n_tr)
# Fraction of attachment edges (test→train) that cross to same-subject train neighbor
att_s_pct = (y_sub_np[ei_att_te[1].numpy()] == y_sub_np[ei_att_te[0].numpy()]).mean() * 100
# Fraction of test nodes that have ≥1 same-subject attachment neighbor
att_same_s = (y_sub_np[n_tr+n_val:].reshape(-1,1) ==
              y_sub_np[ei_att_te[0].numpy()].reshape(n_te, k)).any(1).mean() * 100
print(f"  Train graph: edges={sg['edges']}, mean_deg={sg['mean_deg']:.2f}, "
      f"same_grade={sg['same_grade_pct']:.1f}%, same_subj={sg['same_subj_pct']:.1f}%")
print(f"  Attachment (test): same-subject={att_s_pct:.1f}%  [Hg2]")

results['graphs'] = {
    'strategy1': {**sg, 'attach_subject_homophily_pct': round(float(att_s_pct), 2),
                  'k': int(k), 'attach_edges_val': int(ei_att_val.shape[1]),
                  'attach_edges_test': int(ei_att_te.shape[1])}
}

# ─── G2: τ-sweep on threshold graph ──────────────────────────────────────────
print("\n=== τ-sweep (seed=42, val acc of SAGE on threshold train graph) ===")
tau_results = {}; best_tau, best_tau_acc = TAU_SWEEP_VALS[0], -1.0
for tau in TAU_SWEEP_VALS:
    ei_tau_tr = threshold_train(x_tr_norm, tau)
    if ei_tau_tr.shape[1] == 0:
        tau_results[str(tau)] = 0.; continue
    ei_tau_val = torch.cat([ei_tau_tr, ei_att_val], 1)
    set_seed(42)
    m = SAGEDual()
    m = train_dual_sage(m, x_ro, ei_tau_tr, ei_tau_val, y_sub_ro, y_grd_ro, tr_mask, val_mask)
    m.eval()
    with torch.no_grad(): vs, _ = m(x_ro, ei_tau_val)
    va = (vs[val_mask].argmax(1) == y_sub_ro[val_mask]).float().mean().item() * 100
    st = graph_stats(ei_tau_tr, y_sub_np, y_grd_np, n_tr)
    tau_results[str(tau)] = round(va, 3)
    print(f"  τ={tau}: val={va:.2f}%, edges={st['edges']}")
    if va > best_tau_acc: best_tau_acc = va; best_tau = tau

print(f"  → τ* = {best_tau}  (val_acc={best_tau_acc:.2f}%)")
results['tau']     = float(best_tau)
results['tau_sweep'] = tau_results

# Build final threshold graph
ei_thresh_tr   = threshold_train(x_tr_norm, best_tau)
ei_thresh_test = torch.cat([ei_thresh_tr, ei_att_val, ei_att_te], 1)
st_thresh = graph_stats(ei_thresh_tr, y_sub_np, y_grd_np, n_tr)
results['graphs']['threshold'] = {**st_thresh, 'tau': float(best_tau)}
print(f"  Threshold graph: edges={st_thresh['edges']}, same_subj={st_thresh['same_subj_pct']:.1f}%")

# Feature-only kNN (same k, no subject filter)
ei_feat_tr   = feature_knn_train(x_tr_norm, k)
ei_feat_test = torch.cat([ei_feat_tr, ei_att_val, ei_att_te], 1)
sf = graph_stats(ei_feat_tr, y_sub_np, y_grd_np, n_tr)
results['graphs']['feature_only'] = {**sf, 'k': int(k)}
print(f"  Feature-only kNN: edges={sf['edges']}, same_subj={sf['same_subj_pct']:.1f}%")

# ─── H: GAT heads sweep ────────────────────────────────────────────────────────
print("\n=== GAT heads sweep (seed=42) ===")
best_heads, best_gat_acc = 4, -1.0
for heads in [2, 4, 8]:
    set_seed(42)
    m = GATDual(heads=heads)
    m = train_dual_sage(m, x_ro, ei_tr, ei_for_val, y_sub_ro, y_grd_ro, tr_mask, val_mask)
    m.eval()
    with torch.no_grad(): vs, _ = m(x_ro, ei_for_val)
    va = (vs[val_mask].argmax(1) == y_sub_ro[val_mask]).float().mean().item() * 100
    print(f"  heads={heads}: val={va:.2f}%")
    if va > best_gat_acc: best_gat_acc = va; best_heads = heads
print(f"  → best_heads = {best_heads}")

# ─── I: kNN-Voting k sweep ────────────────────────────────────────────────────
print("\n=== kNN-Voting k sweep (seed=42, val acc) ===")
best_knn_k, best_knn_acc = 9, -1.0
for knn_k in [3, 5, 9, 18]:
    sim_tr_val = x_val_norm @ x_tr_norm.T  # (n_val, n_tr)
    topk = np.argpartition(sim_tr_val, -knn_k, axis=1)[:, -knn_k:]
    pred_s = np.apply_along_axis(lambda x: np.bincount(y_sub_np[x], minlength=NUM_S).argmax(), 1, topk)
    va = (pred_s == y_sub_np[n_tr:n_tr+n_val]).mean() * 100
    print(f"  knn_k={knn_k}: val={va:.2f}%")
    if va > best_knn_acc: best_knn_acc = va; best_knn_k = knn_k
print(f"  → knn_k* = {best_knn_k}")

# ─── J: Main model runs (all 8 models × 5 seeds on Strategy 1 graph) ─────────
print("\n" + "="*60)
print("MAIN MODEL RUNS (5 seeds, Strategy 1 graph, test=1929)")
print("="*60)
all_models = {}

def run_model(name, fn):
    print(f"\n--- {name} ---")
    per = run5seeds_sage(fn, ei_tr, ei_for_test)
    s = summarize(per)
    print(f"  → {s['subj_mean']:.2f}±{s['subj_sd']:.2f}% subj, grade={s['grade_mean']:.2f}%, F1={s['f1_macro']:.3f}")
    all_models[name] = s
    return per

sage_per   = run_model('sage_mean',   lambda: SAGEDual())
attn_per   = run5seeds_sage(lambda: AttnSAGEDual(), ei_tr, ei_for_test, attn_mode=True)
all_models['attn_sage'] = summarize(attn_per)
print(f"  attn_sage: {all_models['attn_sage']['subj_mean']:.2f}±{all_models['attn_sage']['subj_sd']:.2f}%")
gcn_per    = run_model('gcn',         lambda: GCNDual())
gat_per    = run_model('gat',         lambda: GATDual(heads=best_heads))
lstm_per   = run_model('sage_lstm',   lambda: SAGEDualLSTM())
pool_per   = run_model('sage_pool',   lambda: SAGEDual(aggr='max'))

# Fusion MLP (no graph) — Option B for VisualBERT
print("\n--- fusion_mlp (no graph) ---")
mlp_per = []
for seed in SEEDS:
    set_seed(seed)
    m = MLPDual()
    m = train_dual_mlp(m, x_ro, y_sub_ro, y_grd_ro, tr_mask, ei_for_val, val_mask)
    r = eval_mlp(m, x_ro, te_mask, y_sub_ro, y_grd_ro)
    mlp_per.append((r[0], r[1], r[2], r[2], r[2]))
    print(f"    seed={seed}: subj={r[0]:.2f}%, grade={r[1]:.2f}%")
all_models['fusion_mlp'] = summarize(mlp_per)

# Text-only
print("\n--- text_only ---")
text_per = []
for seed in SEEDS:
    set_seed(seed)
    m = MLPDual(in_ch=384)
    m = train_dual_mlp(m, xt_ro, y_sub_ro, y_grd_ro, tr_mask, ei_for_val, val_mask)
    r = eval_mlp(m, xt_ro, te_mask, y_sub_ro, y_grd_ro)
    text_per.append((r[0], r[1], r[2], r[2], r[2]))
    print(f"    seed={seed}: subj={r[0]:.2f}%")
all_models['text_only'] = summarize(text_per)

# Image-only (CLIP, Logistic Regression)
print("\n--- image_only ---")
img_per = []
for seed in SEEDS:
    lr_m = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    lr_m.fit(xi_ro[tr_mask].numpy(), y_sub_ro[tr_mask].numpy())
    pred_s = lr_m.predict(xi_ro[te_mask].numpy())
    pred_g = np.zeros_like(pred_s)  # placeholder (LR doesn't predict grade)
    acc_s = (pred_s == y_sub_ro[te_mask].numpy()).mean() * 100
    # For grade: use separate LR
    lr_g = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
    lr_g.fit(xi_ro[tr_mask].numpy(), y_grd_ro[tr_mask].numpy())
    pred_g = lr_g.predict(xi_ro[te_mask].numpy())
    acc_g = (pred_g == y_grd_ro[te_mask].numpy()).mean() * 100
    f1_s  = f1_score(y_sub_ro[te_mask].numpy(), pred_s, average='macro', zero_division=0)
    img_per.append((acc_s, acc_g, f1_s, f1_s, f1_s))
    print(f"    seed={seed}: subj={acc_s:.2f}%")
all_models['image_only'] = summarize(img_per)

# kNN-Voting
print("\n--- knn_voting ---")
knn_per = []
for seed in SEEDS:
    sim = x_te_norm @ x_tr_norm.T
    tops = np.argpartition(sim, -best_knn_k, axis=1)[:, -best_knn_k:]
    pred_s = np.apply_along_axis(
        lambda x: np.bincount(y_sub_np[:n_tr][x], minlength=NUM_S).argmax(), 1, tops)
    pred_g = np.apply_along_axis(
        lambda x: np.bincount(y_grd_np[:n_tr][x], minlength=NUM_G).argmax(), 1, tops)
    acc_s = (pred_s == y_sub_np[n_tr+n_val:]).mean() * 100
    acc_g = (pred_g == y_grd_np[n_tr+n_val:]).mean() * 100
    f1_s  = f1_score(y_sub_np[n_tr+n_val:], pred_s, average='macro', zero_division=0)
    knn_per.append((acc_s, acc_g, f1_s, f1_s, f1_s))
    print(f"    seed={seed}: subj={acc_s:.2f}%")
all_models['knn_voting'] = summarize(knn_per)

results['models'] = all_models

# ─── K: Graph comparison (4b feature-only, 4c threshold) ─────────────────────
print("\n=== Graph comparison (feature-only, threshold) ===")
graph_comp = {}
for gname, g_ei_test in [('feature_only', ei_feat_test), ('threshold', ei_thresh_test)]:
    g_ei_tr = ei_feat_tr if gname == 'feature_only' else ei_thresh_tr
    g_ei_val = torch.cat([g_ei_tr, ei_att_val], 1)
    per = []
    print(f"\n  {gname}:")
    for seed in SEEDS:
        set_seed(seed)
        m = SAGEDual()
        m = train_dual_sage(m, x_ro, g_ei_tr, g_ei_val, y_sub_ro, y_grd_ro, tr_mask, val_mask)
        r = eval_dual(m, x_ro, g_ei_test, te_mask, y_sub_ro, y_grd_ro)
        per.append(r[:5])
        print(f"    seed={seed}: subj={r[0]:.2f}%")
    s = summarize(per)
    graph_comp[gname] = s
    print(f"  {gname}: {s['subj_mean']:.2f}±{s['subj_sd']:.2f}%")

results['graph_comparison'] = graph_comp

# ─── L: Random edge control (4e) ──────────────────────────────────────────────
print("\n=== Random edge control ===")
n_rand_edges = ei_tr.shape[1]
rand_per = []
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    src_r = rng.randint(0, n_tr, n_rand_edges); dst_r = rng.randint(0, n_tr, n_rand_edges)
    ei_rand = torch.tensor(np.stack([src_r, dst_r]), dtype=torch.long)
    ei_rand_val  = torch.cat([ei_rand, ei_att_val], 1)
    ei_rand_test = torch.cat([ei_rand, ei_att_val, ei_att_te], 1)
    set_seed(seed)
    m = SAGEDual()
    m = train_dual_sage(m, x_ro, ei_rand, ei_rand_val, y_sub_ro, y_grd_ro, tr_mask, val_mask)
    r = eval_dual(m, x_ro, ei_rand_test, te_mask, y_sub_ro, y_grd_ro)
    rand_per.append(r[:5])
    print(f"  seed={seed}: subj={r[0]:.2f}%")
rand_subjs = [r[0] for r in rand_per]
results['random_edge'] = {
    'per_seed': [round(s, 4) for s in rand_subjs],
    'mean': round(float(np.mean(rand_subjs)), 4),
    'sd':   round(float(np.std(rand_subjs)), 4),
    'n_edges': int(n_rand_edges),
}
print(f"  Random control: {np.mean(rand_subjs):.2f}±{np.std(rand_subjs):.2f}%")

# ─── M: Upper bounds (label-informed, 4d) ─────────────────────────────────────
print("\n=== Upper bounds (label-informed, eval on new test 1929) ===")
ei_orig_raw = g2['edge_index'].long()
# Remap to reordered indices
node_map = torch.zeros(N, dtype=torch.long)
for new_i, old_i in enumerate(reorder): node_map[old_i] = new_i
ei_orig_ro = node_map[ei_orig_raw]

def intra_class42_knn_all(x_np_norm, y_comp_np, k_meta=9):
    all_src, all_dst = [], []
    for c in np.unique(y_comp_np):
        idx = np.where(y_comp_np == c)[0]
        if len(idx) < 2: continue
        sim = x_np_norm[idx] @ x_np_norm[idx].T; np.fill_diagonal(sim, -1.)
        ki = min(k_meta, len(idx)-1)
        tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
        all_src.append(idx[np.repeat(np.arange(len(idx)), ki)])
        all_dst.append(idx[tops.ravel()])
    return torch.tensor(np.stack([np.concatenate(all_src), np.concatenate(all_dst)]), dtype=torch.long)

y_comp_ro = (y_sub_ro * NUM_G + y_grd_ro).long()
x_all_norm = x_ro.numpy() / (np.linalg.norm(x_ro.numpy(), axis=1, keepdims=True) + 1e-8)
ei_meta = intra_class42_knn_all(x_all_norm, y_comp_ro.numpy(), k_meta=9)
ei_intra_s = node_map[torch.tensor(
    np.concatenate([np.where(y_sub_np[:] == s)[0] for s in np.unique(y_sub_np)]),
    dtype=torch.long)]  # placeholder

upper_bounds = {}
for ub_name, ei_ub in [('semi_transductive', ei_orig_ro), ('metadata', ei_meta)]:
    per_ub = []
    print(f"\n  {ub_name} (edges={ei_ub.shape[1]}):")
    for seed in SEEDS:
        set_seed(seed)
        m = SAGEDual()
        # Semi-transductive: all nodes in graph, loss on tr_mask only
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        best_vl, pat, best_st = float('inf'), 0, None
        for ep in range(MAX_EP):
            m.train(); opt.zero_grad()
            ls, lg = m(x_ro, ei_ub)
            loss = F.cross_entropy(ls[tr_mask], y_sub_ro[tr_mask]) + \
                   F.cross_entropy(lg[tr_mask], y_grd_ro[tr_mask])
            loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vs, vg = m(x_ro, ei_ub)
                vl = (F.cross_entropy(vs[val_mask], y_sub_ro[val_mask]) +
                      F.cross_entropy(vg[val_mask], y_grd_ro[val_mask])).item()
            if vl < best_vl - 1e-4: best_vl = vl; pat = 0; best_st = {k2: v.clone() for k2, v in m.state_dict().items()}
            else:
                pat += 1
                if pat >= PAT: break
        if best_st: m.load_state_dict(best_st)
        r = eval_dual(m, x_ro, ei_ub, te_mask, y_sub_ro, y_grd_ro)
        per_ub.append(r[:5]); print(f"    seed={seed}: subj={r[0]:.2f}%")
    s = summarize(per_ub)
    upper_bounds[ub_name] = s
    print(f"  {ub_name}: {s['subj_mean']:.2f}±{s['subj_sd']:.2f}%")

results['upper_bounds'] = upper_bounds

# ─── N: Statistics ────────────────────────────────────────────────────────────
print("\n=== Statistics ===")
ours_subjs = np.array(all_models['sage_mean']['per_seed_subj'])
stats_vs_ours = {}
baselines = ['attn_sage', 'gcn', 'gat', 'fusion_mlp', 'knn_voting', 'text_only', 'image_only']
n_comp = len(baselines)
for b in baselines:
    b_subjs = np.array(all_models[b]['per_seed_subj'])
    diff = ours_subjs - b_subjs
    t_val, p_val = stats.ttest_rel(ours_subjs, b_subjs)
    d = diff.mean() / (diff.std() + 1e-9)  # Cohen's d paired
    se = diff.std() / np.sqrt(5)
    t_crit = stats.t.ppf(0.975, df=4)
    ci = [round(diff.mean() - t_crit*se, 3), round(diff.mean() + t_crit*se, 3)]
    p_bon = min(1.0, float(p_val) * n_comp)
    stats_vs_ours[b] = {
        'delta_pp': round(float(diff.mean()), 3),
        'ci95': ci, 't': round(float(t_val), 3), 'p': round(float(p_val), 4),
        'p_bonferroni': round(p_bon, 4), 'd': round(float(d), 3)
    }
    sig = "***" if p_bon < 0.0125 else ("*" if float(p_val) < 0.05 else "ns")
    print(f"  ours vs {b}: Δ={diff.mean():+.2f}pp, p={p_val:.4f}, p_bon={p_bon:.4f} {sig}, d={d:.2f}")

results['stats_vs_ours'] = stats_vs_ours

# ─── O: Per-class Table 1 (Ours, seed=42) ─────────────────────────────────────
print("\n=== Table 1 per-class (seed=42, sage_mean, test) ===")
set_seed(42)
m_t1 = SAGEDual()
m_t1 = train_dual_sage(m_t1, x_ro, ei_tr, ei_for_val, y_sub_ro, y_grd_ro, tr_mask, val_mask)
r_t1 = eval_dual(m_t1, x_ro, ei_for_test, te_mask, y_sub_ro, y_grd_ro)
pred_s_42 = r_t1[5]; true_s = y_sub_ro[te_mask]
prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
    true_s.numpy(), pred_s_42.numpy(), labels=list(range(NUM_S)), zero_division=0)
table1 = {}
for i, name in enumerate(le_s.classes_):
    table1[name] = {'prec': round(float(prec_c[i]),3), 'rec': round(float(rec_c[i]),3),
                    'f1': round(float(f1_c[i]),3), 'n': int(sup_c[i])}
    print(f"  {name:<45} P={prec_c[i]:.3f} R={rec_c[i]:.3f} F1={f1_c[i]:.3f} N={sup_c[i]}")
macro_prf = {'prec': round(float(prec_c.mean()),3), 'rec': round(float(rec_c.mean()),3),
             'f1': round(float(f1_c.mean()),3)}
print(f"  Macro: {macro_prf}")
results['table1_per_class'] = table1
results['table1_macro'] = macro_prf

# ─── P: Calibration (temperature scaling, ECE) ───────────────────────────────
print("\n=== Calibration ===")
# Get val and test logits from m_t1 (seed=42 SAGE mean)
m_t1.eval()
with torch.no_grad():
    vs_t1, _ = m_t1(x_ro, ei_for_val)
    vlogits = vs_t1[val_mask].numpy()
    vtrue   = y_sub_ro[val_mask].numpy()
    ts_t1, _ = m_t1(x_ro, ei_for_test)
    tlogits = ts_t1[te_mask].numpy()
    ttrue   = y_sub_ro[te_mask].numpy()

def nll_fn(T, logits, labels):
    t = torch.tensor(logits / T); return F.cross_entropy(t, torch.tensor(labels)).item()

res_T = minimize_scalar(lambda T: nll_fn(T, vlogits, vtrue), bounds=(0.1, 10.), method='bounded')
T_opt = float(res_T.x)

def ece(logits, labels, n_bins=15):
    probs = torch.tensor(logits).softmax(1).numpy()
    conf  = probs.max(1); pred  = probs.argmax(1); acc = (pred == labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins+1); ece_val = 0.
    for i in range(n_bins):
        in_bin = (conf >= bin_edges[i]) & (conf < bin_edges[i+1])
        if in_bin.sum() > 0:
            ece_val += in_bin.mean() * abs(acc[in_bin].mean() - conf[in_bin].mean())
    return float(ece_val)

ece_before = ece(tlogits, ttrue)
ece_after  = ece(tlogits / T_opt, ttrue)
print(f"  T* = {T_opt:.3f},  ECE before = {ece_before:.4f},  ECE after = {ece_after:.4f}")
results['calibration'] = {'T': round(T_opt, 4), 'ece_before': round(ece_before, 4),
                          'ece_after': round(ece_after, 4)}

# ─── Q: Confusion matrix + error analysis ─────────────────────────────────────
print("\n=== Error analysis (seed=42, sage_mean) ===")
cm = confusion_matrix(true_s.numpy(), pred_s_42.numpy(), labels=list(range(NUM_S)))
errors_idx = np.where(pred_s_42.numpy() != true_s.numpy())[0]

# Mean CLIP feature per class
class_centers = np.stack([x_te_norm[true_s.numpy() == s].mean(0) for s in range(NUM_S)])
class_sims = class_centers @ class_centers.T

# Predefined interdisciplinary pairs (thematic similarity)
interdisciplinary_pairs = {(0,1),(1,0),(4,13),(13,4),(3,7),(7,3),(2,6),(6,2)}  # rough examples

err_visual, err_inter, err_noise, err_imbal = 0, 0, 0, 0
probs_te = torch.tensor(tlogits).softmax(1).numpy()
rare_classes = set(i for i in range(NUM_S) if sup_c[i] < 80)

for i in errors_idx:
    pred = pred_s_42[i].item(); true = true_s[i].item()
    conf = probs_te[i].max()
    if class_sims[pred, true] > 0.85: err_visual += 1        # [E1]
    elif (pred, true) in interdisciplinary_pairs: err_inter += 1  # [E2]
    elif conf > 0.85: err_noise += 1                          # [E3]
    elif true in rare_classes: err_imbal += 1                 # [E4]

n_err = len(errors_idx)
if n_err > 0:
    e_pct = {'visual_sim': round(err_visual/n_err*100,1), 'interdisciplinary': round(err_inter/n_err*100,1),
             'noise': round(err_noise/n_err*100,1), 'imbalance': round(err_imbal/n_err*100,1)}
else:
    e_pct = {'visual_sim': 0., 'interdisciplinary': 0., 'noise': 0., 'imbalance': 0.}
print(f"  Total errors: {n_err} / {n_te}  ({n_err/n_te*100:.1f}%)")
print(f"  Error types: {e_pct}")
results['errors_pct'] = e_pct
results['confusion_matrix'] = cm.tolist()

# t-SNE (compute, save coords)
print("\n=== t-SNE (embeddings, seed=42) ===")
m_t1.eval()
with torch.no_grad(): emb = m_t1.embed(x_ro, ei_for_test)[te_mask].numpy()
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000).fit_transform(emb)
results['tsne'] = {'note': 'Saved 2D coords for 1929 test nodes',
                   'shape': [int(tsne_2d.shape[0]), 2]}
np.save(r'd:\SPNC_gnnclassifier-main\tsne_coords.npy', tsne_2d)
print(f"  t-SNE saved to tsne_coords.npy  shape={tsne_2d.shape}")

# ─── R: Aggregator ablation (Ag1/Ag2/Ag3 on Strategy 1 graph) ────────────────
print("\n=== Aggregator ablation ===")
agg_results = {}
for agg_name, fn in [('mean_Ag1', lambda: SAGEDual(aggr='mean')),
                     ('pool_Ag2', lambda: SAGEDual(aggr='max')),
                     ('lstm_Ag3', lambda: SAGEDualLSTM())]:
    print(f"\n  {agg_name}:")
    per_agg = run5seeds_sage(fn, ei_tr, ei_for_test)
    agg_results[agg_name] = summarize(per_agg)
    print(f"  → {agg_results[agg_name]['subj_mean']:.2f}±{agg_results[agg_name]['subj_sd']:.2f}%")
results['aggregator_ablation'] = agg_results

# ─── S: Finalize and save ─────────────────────────────────────────────────────
results['env'] = {
    'torch': torch.__version__,
    'python': sys.version.split()[0],
    'visualbert_note': 'Option B: renamed to Fusion MLP (MLP on 896-dim concat), citation [10] removed',
    'hp': {'lr': LR, 'wd': WD, 'hidden': H, 'dropout': DR, 'patience': PAT, 'max_ep': MAX_EP},
    'batch': 'full-batch GNN (6750 train nodes), full-batch MLP',
    'gat_heads': int(best_heads),
    'knn_voting_k': int(best_knn_k),
}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n\n{'='*60}")
print(f"Saved: {OUT}")
print(f"{'='*60}")

# ─── Final summary table ──────────────────────────────────────────────────────
print("\n=== FINAL SUMMARY (test=1929 nodes) ===")
print(f"{'Model':<25} {'SubjAcc':>10} {'SD':>6} {'Grade':>7} {'F1':>6}")
print("-" * 60)
key_order = ['sage_mean','attn_sage','gcn','gat','fusion_mlp','knn_voting','text_only','image_only']
for k2 in key_order:
    d = all_models[k2]
    print(f"  {k2:<23} {d['subj_mean']:>8.2f}  {d['subj_sd']:>4.2f}  {d['grade_mean']:>6.2f}  {d['f1_macro']:>5.3f}")
print(f"\n  Random edge ctrl:       {results['random_edge']['mean']:>6.2f} ± {results['random_edge']['sd']:.2f}")
print("\n  Upper bounds (label-informed):")
for ub, v in results['upper_bounds'].items():
    print(f"    {ub:<25} {v['subj_mean']:>6.2f} ± {v['subj_sd']:.2f}")
