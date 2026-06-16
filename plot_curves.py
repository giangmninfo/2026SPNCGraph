"""
plot_curves.py — Training/Validation Loss & Accuracy curves
Fusion MLP + GraphSAGE Mean, seed=42
Output: training_curves.png
"""
import sys, io, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import os; os.environ['PYTHONHASHSEED'] = '42'

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import SAGEConv

# ── Config ───────────────────────────────────────────────────────────────────
BASE  = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
SPLIT = r'd:\SPNC_gnnclassifier-main\split_indices.json'
OUT   = r'd:\SPNC_gnnclassifier-main\training_curves.png'

SEED = 42
LR, WD, H, DR, PAT, MAX_EP = 1e-3, 0.0, 128, 0.3, 20, 200
NUM_S, NUM_G = 14, 3

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    import random; random.seed(s)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')
le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub = torch.tensor(le_s.fit_transform(meta['Tên môn'].values), dtype=torch.long)
y_grd = torch.tensor(le_g.fit_transform(meta['Lớp'].values),    dtype=torch.long)
x_full = g2['x'].float(); N = len(y_sub)

with open(SPLIT, encoding='utf-8') as f:
    sp = json.load(f)
tr_i  = np.array(sp['train']); val_i = np.array(sp['val']); te_i = np.array(sp['test'])
n_tr, n_val, n_te = len(tr_i), len(val_i), len(te_i)

reorder  = np.concatenate([tr_i, val_i, te_i])
x_ro     = x_full[reorder]
y_sub_ro = y_sub[reorder]; y_grd_ro = y_grd[reorder]

tr_mask  = torch.zeros(N, dtype=torch.bool); tr_mask[:n_tr]            = True
val_mask = torch.zeros(N, dtype=torch.bool); val_mask[n_tr:n_tr+n_val] = True
te_mask  = torch.zeros(N, dtype=torch.bool); te_mask[n_tr+n_val:]      = True

x_np   = x_ro.numpy()
x_norm = (x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
x_tr_norm  = x_norm[:n_tr]
x_val_norm = x_norm[n_tr:n_tr+n_val]

# ── Graph ──────────────────────────────────────────────────────────────────────
def intra_subj_knn(x_norm_tr, y_sub_tr_np, k):
    all_src, all_dst = [], []
    for s in np.unique(y_sub_tr_np):
        idx = np.where(y_sub_tr_np == s)[0]
        if len(idx) < 2: continue
        sim = x_norm_tr[idx] @ x_norm_tr[idx].T; np.fill_diagonal(sim, -1.)
        ki = min(k, len(idx)-1)
        tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
        all_src.append(idx[np.repeat(np.arange(len(idx)), ki)])
        all_dst.append(idx[tops.ravel()])
    return torch.tensor(np.stack([np.concatenate(all_src), np.concatenate(all_dst)]), dtype=torch.long)

def cosine_knn(x_norm_q, x_norm_ref, k, offset_dst):
    sim = x_norm_q @ x_norm_ref.T
    ki  = min(k, x_norm_ref.shape[0])
    tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
    n_q  = x_norm_q.shape[0]
    dst  = np.repeat(np.arange(n_q) + offset_dst, ki)
    return torch.tensor(np.stack([tops.ravel(), dst]), dtype=torch.long)

y_sub_np = y_sub_ro.numpy()
ei_tr      = intra_subj_knn(x_tr_norm, y_sub_np[:n_tr], 6)
ei_att_val = cosine_knn(x_val_norm, x_tr_norm, 6, n_tr)
ei_for_val = torch.cat([ei_tr, ei_att_val], 1)

# ── Models ─────────────────────────────────────────────────────────────────────
class SAGEDual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = SAGEConv(896, H); self.conv2 = SAGEConv(H, H)
        self.head_s = nn.Linear(H, NUM_S); self.head_g = nn.Linear(H, NUM_G)
    def forward(self, x, ei):
        h = F.dropout(F.relu(self.conv1(x, ei)), DR, self.training)
        h = F.relu(self.conv2(h, ei))
        return self.head_s(h), self.head_g(h)

class MLPDual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net    = nn.Sequential(nn.Linear(896, H*2), nn.ReLU(), nn.Dropout(DR),
                                    nn.Linear(H*2, H), nn.ReLU())
        self.head_s = nn.Linear(H, NUM_S); self.head_g = nn.Linear(H, NUM_G)
    def forward(self, x):
        h = self.net(x)
        return self.head_s(h), self.head_g(h)

# ── Training with history recording ───────────────────────────────────────────
def train_sage_history(seed):
    set_seed(seed)
    model = SAGEDual()
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    hist  = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_vl, pat, best_ep, best_st = float('inf'), 0, 0, None

    for ep in range(MAX_EP):
        model.train(); opt.zero_grad()
        ls, lg = model(x_ro, ei_tr)
        tr_loss = F.cross_entropy(ls[tr_mask], y_sub_ro[tr_mask]) + \
                  F.cross_entropy(lg[tr_mask], y_grd_ro[tr_mask])
        tr_loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            ls_f, lg_f = model(x_ro, ei_for_val)
            vl = (F.cross_entropy(ls_f[val_mask], y_sub_ro[val_mask]) +
                  F.cross_entropy(lg_f[val_mask], y_grd_ro[val_mask])).item()
            tr_acc = (ls_f[tr_mask].argmax(1) == y_sub_ro[tr_mask]).float().mean().item()
            va_acc = (ls_f[val_mask].argmax(1) == y_sub_ro[val_mask]).float().mean().item()

        hist['train_loss'].append(tr_loss.item())
        hist['val_loss'].append(vl)
        hist['train_acc'].append(tr_acc * 100)
        hist['val_acc'].append(va_acc * 100)

        if vl < best_vl - 1e-4:
            best_vl = vl; pat = 0; best_ep = ep
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PAT: break

    if best_st: model.load_state_dict(best_st)
    best_val_acc = hist['val_acc'][best_ep]
    stop_ep = len(hist['train_loss'])
    print(f"  GraphSAGE: stopped ep={stop_ep}, best_ep={best_ep+1}, best_val={best_val_acc:.2f}%")
    return hist, best_ep, stop_ep, best_val_acc

def train_mlp_history(seed):
    set_seed(seed)
    model = MLPDual()
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    hist  = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_vl, pat, best_ep, best_st = float('inf'), 0, 0, None

    for ep in range(MAX_EP):
        model.train(); opt.zero_grad()
        ls, lg = model(x_ro[tr_mask])
        tr_loss = F.cross_entropy(ls, y_sub_ro[tr_mask]) + \
                  F.cross_entropy(lg, y_grd_ro[tr_mask])
        tr_loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            ls_tr, _ = model(x_ro[tr_mask])
            ls_vl, lg_vl = model(x_ro[val_mask])
            vl = (F.cross_entropy(ls_vl, y_sub_ro[val_mask]) +
                  F.cross_entropy(lg_vl, y_grd_ro[val_mask])).item()
            tr_acc = (ls_tr.argmax(1) == y_sub_ro[tr_mask]).float().mean().item()
            va_acc = (ls_vl.argmax(1) == y_sub_ro[val_mask]).float().mean().item()

        hist['train_loss'].append(tr_loss.item())
        hist['val_loss'].append(vl)
        hist['train_acc'].append(tr_acc * 100)
        hist['val_acc'].append(va_acc * 100)

        if vl < best_vl - 1e-4:
            best_vl = vl; pat = 0; best_ep = ep
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PAT: break

    if best_st: model.load_state_dict(best_st)
    best_val_acc = hist['val_acc'][best_ep]
    stop_ep = len(hist['train_loss'])
    print(f"  Fusion MLP: stopped ep={stop_ep}, best_ep={best_ep+1}, best_val={best_val_acc:.2f}%")
    return hist, best_ep, stop_ep, best_val_acc

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training GraphSAGE Mean (seed=42)...")
sage_hist, sage_best_ep, sage_stop, sage_best_val = train_sage_history(SEED)

print("Training Fusion MLP (seed=42)...")
mlp_hist,  mlp_best_ep,  mlp_stop,  mlp_best_val  = train_mlp_history(SEED)

# ── Plot ──────────────────────────────────────────────────────────────────────
BLUE  = '#1f77b4'
RED   = '#d62728'
GRAY  = '#888888'
GREEN = '#2ca02c'

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.subplots_adjust(hspace=0.38, wspace=0.28)

def smooth(arr, w=3):
    """Simple moving average."""
    out = np.convolve(arr, np.ones(w)/w, mode='same')
    out[:w//2] = arr[:w//2]; out[-(w//2):] = arr[-(w//2):]
    return out

def plot_pair(ax_loss, ax_acc, hist, best_ep, stop_ep, best_val, title):
    epochs = np.arange(1, stop_ep + 1)
    tl = np.array(hist['train_loss']); vl = np.array(hist['val_loss'])
    ta = np.array(hist['train_acc']);  va = np.array(hist['val_acc'])

    # Loss panel
    ax_loss.plot(epochs, smooth(tl, 3), color=BLUE, lw=1.6, label='Train Loss')
    ax_loss.plot(epochs, smooth(vl, 3), color=RED,  lw=1.6, label='Val Loss', alpha=0.85)
    ax_loss.axvline(best_ep + 1, color=GRAY, lw=1.2, ls='--', alpha=0.7)
    ax_loss.text(best_ep + 2, ax_loss.get_ylim()[1] * 0.92,
                 'Early\nStop', color=GRAY, fontsize=7.5, va='top')
    ax_loss.set_xlabel('Epoch', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.set_title(f'(a) Loss Curves — {title}', fontsize=10.5, pad=6)
    ax_loss.legend(fontsize=8.5, framealpha=0.6)
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)
    ax_loss.set_xlim(1, stop_ep + 1)

    # Accuracy panel
    ax_acc.plot(epochs, smooth(ta, 3), color=BLUE, lw=1.6, label='Train Acc')
    ax_acc.plot(epochs, smooth(va, 3), color=RED,  lw=1.6, label='Val Acc', alpha=0.85)
    ax_acc.axvline(best_ep + 1, color=GRAY, lw=1.2, ls='--', alpha=0.7)
    # Best annotation (green box)
    ax_acc.annotate(f'Best: {best_val:.2f}%',
                    xy=(best_ep + 1, best_val),
                    xytext=(best_ep + 4, best_val - 4),
                    fontsize=8.5, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=GREEN, alpha=0.85, edgecolor='none'),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))
    ax_acc.set_xlabel('Epoch', fontsize=10)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=10)
    ax_acc.set_title(f'(b) Accuracy Curves — {title}', fontsize=10.5, pad=6)
    ax_acc.legend(fontsize=8.5, framealpha=0.6)
    ax_acc.spines['top'].set_visible(False)
    ax_acc.spines['right'].set_visible(False)
    ax_acc.set_xlim(1, stop_ep + 1)
    ax_acc.set_ylim(30, 102)

# Row 0: Fusion MLP
plot_pair(axes[0, 0], axes[0, 1],
          mlp_hist,  mlp_best_ep,  mlp_stop,  mlp_best_val,
          'Fusion MLP')

# Row 1: GraphSAGE Mean
plot_pair(axes[1, 0], axes[1, 1],
          sage_hist, sage_best_ep, sage_stop, sage_best_val,
          'GraphSAGE Mean')

plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {OUT}")
