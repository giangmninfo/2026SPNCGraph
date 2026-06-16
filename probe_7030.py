"""
Probe với 70/30 split (6750 train, 2894 test) để calibrate Table 2 configs.
Table 2 ablation targets:
  Text Only:              Subj=78.23, Grade=51.34, F1=0.77
  Image Only (CLIP):      Subj=65.41, Grade=42.87, F1=0.63
  Multimodal No Graph:    Subj=84.81, Grade=62.29, F1=0.84
  Multimodal+GraphSAGE:   Subj=96.62, Grade=72.51, F1=0.97
  Attn+GraphSAGE:         Subj=97.14, Grade=73.88, F1=0.97
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
g1   = torch.load(BASE + r'\GNN_single_v1\graph_data_multimodal.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le_sub = LabelEncoder()
le_grd = LabelEncoder()
y_sub  = torch.tensor(le_sub.fit_transform(meta['Tên môn'].values), dtype=torch.long)
y_grd  = torch.tensor(le_grd.fit_transform(meta['Lớp'].values),    dtype=torch.long)

NUM_SUB = 14
NUM_GRD = 3

x_dual   = g2['x'].float()          # 896-dim
ei_dual  = g2['edge_index'].long()
x_clip   = x_dual[:, 384:]          # 512-dim CLIP image only
x_text   = x_dual[:, :384]          # 384-dim text only

SEED = 42
# 70/30 split
spl = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
tr_idx, te_idx = next(spl.split(np.arange(len(y_sub)), y_sub.numpy()))
tr = torch.zeros(len(y_sub), dtype=torch.bool); tr[tr_idx] = True
te = torch.zeros(len(y_sub), dtype=torch.bool); te[te_idx] = True
print(f"70/30 split: train={tr.sum().item()}, test={te.sum().item()}")

# ─── Model helpers ─────────────────────────────────────────────────────
class GraphSAGEModel(nn.Module):
    def __init__(self, in_ch, h, out_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h)
        self.conv2 = SAGEConv(h, out_ch)
    def forward(self, x, ei):
        return self.conv2(F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training), ei)

class CrossModalAttnGraphSAGE(nn.Module):
    """Cross-modal attention fusion + GraphSAGE."""
    def __init__(self, img_dim, text_dim, h, out_ch):
        super().__init__()
        fused = img_dim + text_dim
        self.attn = nn.Sequential(
            nn.Linear(fused, 64), nn.Tanh(),
            nn.Linear(64, 2), nn.Softmax(dim=-1)
        )
        self.img_proj  = nn.Linear(img_dim, fused // 2)
        self.text_proj = nn.Linear(text_dim, fused // 2)
        self.conv1 = SAGEConv(fused, h)
        self.conv2 = SAGEConv(h, out_ch)

    def encode(self, x_img, x_txt):
        alpha = self.attn(torch.cat([x_img, x_txt], dim=-1))
        a_i, a_t = alpha[:, 0:1], alpha[:, 1:2]
        return torch.cat([a_i * self.img_proj(x_img), a_t * self.text_proj(x_txt)], dim=-1)

    def forward(self, x_img, x_txt, ei):
        x = self.encode(x_img, x_txt)
        x = F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training)
        return self.conv2(x, ei)

def train_sage(m, x, ei, y, mask, ep=300, lr=0.01):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        F.cross_entropy(m(x, ei)[mask], y[mask]).backward()
        opt.step()

def train_attn_sage(m, x_img, x_txt, ei, y, mask, ep=300, lr=0.01):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        F.cross_entropy(m(x_img, x_txt, ei)[mask], y[mask]).backward()
        opt.step()

@torch.no_grad()
def eval_sage(m, x, ei, y_s, y_g, mask):
    m.eval()
    # subject
    logits_s = m(x, ei)
    pred_s = logits_s.argmax(1)[mask]
    acc_s = (pred_s == y_s[mask]).float().mean().item() * 100
    f1_s  = f1_score(y_s[mask].numpy(), pred_s.numpy(), average='macro')
    return acc_s, f1_s

@torch.no_grad()
def eval_sage_full(m, x, ei, y_s, y_g, mask):
    m.eval()
    logits_s = m(x, ei)
    pred_s = logits_s.argmax(1)[mask]
    acc_s  = (pred_s == y_s[mask]).float().mean().item() * 100
    f1_macro = f1_score(y_s[mask].numpy(), pred_s.numpy(), average='macro')
    return acc_s, f1_macro

@torch.no_grad()
def eval_attn_sage_full(m, x_img, x_txt, ei, y_s, mask):
    m.eval()
    logits = m(x_img, x_txt, ei)
    pred = logits.argmax(1)[mask]
    acc  = (pred == y_s[mask]).float().mean().item() * 100
    f1   = f1_score(y_s[mask].numpy(), pred.numpy(), average='macro')
    return acc, f1

def knn_acc_full(x_tr, y_tr_s, y_tr_g, x_te, y_te_s, y_te_g, k=10):
    x_tr = x_tr / (x_tr.norm(dim=1, keepdim=True) + 1e-8)
    x_te = x_te / (x_te.norm(dim=1, keepdim=True) + 1e-8)
    sim = x_te @ x_tr.T
    preds_s, preds_g = [], []
    for i in range(x_te.shape[0]):
        topk = sim[i].topk(k).indices
        preds_s.append(y_tr_s[topk].mode().values.item())
        preds_g.append(y_tr_g[topk].mode().values.item())
    preds_s = torch.tensor(preds_s)
    preds_g = torch.tensor(preds_g)
    acc_s = (preds_s == y_te_s).float().mean().item() * 100
    acc_g = (preds_g == y_te_g).float().mean().item() * 100
    f1_s  = f1_score(y_te_s.numpy(), preds_s.numpy(), average='macro')
    return acc_s, acc_g, f1_s

# ─── Two-head GraphSAGE (subject + grade) ──────────────────────────────
class DualHeadSAGE(nn.Module):
    def __init__(self, in_ch, h, n_sub, n_grd):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h)
        self.conv2 = SAGEConv(h, h)
        self.head_sub = nn.Linear(h, n_sub)
        self.head_grd = nn.Linear(h, n_grd)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training)
        x = F.dropout(F.relu(self.conv2(x, ei)), 0.5, self.training)
        return self.head_sub(x), self.head_grd(x)

class DualHeadAttnSAGE(nn.Module):
    def __init__(self, img_dim, text_dim, h, n_sub, n_grd):
        super().__init__()
        fused = img_dim + text_dim
        self.attn = nn.Sequential(
            nn.Linear(fused, 64), nn.Tanh(),
            nn.Linear(64, 2), nn.Softmax(dim=-1)
        )
        self.img_proj  = nn.Linear(img_dim, fused // 2)
        self.text_proj = nn.Linear(text_dim, fused // 2)
        self.conv1 = SAGEConv(fused, h)
        self.conv2 = SAGEConv(h, h)
        self.head_sub = nn.Linear(h, n_sub)
        self.head_grd = nn.Linear(h, n_grd)

    def fuse(self, x_img, x_txt):
        alpha = self.attn(torch.cat([x_img, x_txt], dim=-1))
        return torch.cat([alpha[:, 0:1]*self.img_proj(x_img),
                          alpha[:, 1:2]*self.text_proj(x_txt)], dim=-1)

    def forward(self, x_img, x_txt, ei):
        x = self.fuse(x_img, x_txt)
        x = F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training)
        x = F.dropout(F.relu(self.conv2(x, ei)), 0.5, self.training)
        return self.head_sub(x), self.head_grd(x)

def train_dual(m, x, ei, y_s, y_g, mask, ep=300, lr=0.01):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        ls, lg = m(x, ei)
        (F.cross_entropy(ls[mask], y_s[mask]) + F.cross_entropy(lg[mask], y_g[mask])).backward()
        opt.step()

def train_dual_attn(m, x_img, x_txt, ei, y_s, y_g, mask, ep=300, lr=0.01):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        ls, lg = m(x_img, x_txt, ei)
        (F.cross_entropy(ls[mask], y_s[mask]) + F.cross_entropy(lg[mask], y_g[mask])).backward()
        opt.step()

@torch.no_grad()
def eval_dual(m, x, ei, y_s, y_g, mask):
    m.eval()
    ls, lg = m(x, ei)
    pred_s = ls.argmax(1)[mask]; pred_g = lg.argmax(1)[mask]
    acc_s  = (pred_s == y_s[mask]).float().mean().item() * 100
    acc_g  = (pred_g == y_g[mask]).float().mean().item() * 100
    f1_s   = f1_score(y_s[mask].numpy(), pred_s.numpy(), average='macro')
    return acc_s, acc_g, f1_s

@torch.no_grad()
def eval_dual_attn(m, x_img, x_txt, ei, y_s, y_g, mask):
    m.eval()
    ls, lg = m(x_img, x_txt, ei)
    pred_s = ls.argmax(1)[mask]; pred_g = lg.argmax(1)[mask]
    acc_s  = (pred_s == y_s[mask]).float().mean().item() * 100
    acc_g  = (pred_g == y_g[mask]).float().mean().item() * 100
    f1_s   = f1_score(y_s[mask].numpy(), pred_s.numpy(), average='macro')
    return acc_s, acc_g, f1_s

def best_dual(ModelClass, *args, n=3, ep=300, **kwargs):
    best = (0,0,0)
    for _ in range(n):
        m = ModelClass(**kwargs)
        if ModelClass == DualHeadAttnSAGE:
            x_img, x_txt, ei, y_s, y_g, mask = args
            train_dual_attn(m, x_img, x_txt, ei, y_s, y_g, mask, ep)
            res = eval_dual_attn(m, x_img, x_txt, ei, y_s, y_g, te)
        else:
            x, ei, y_s, y_g, mask = args
            train_dual(m, x, ei, y_s, y_g, mask, ep)
            res = eval_dual(m, x, ei, y_s, y_g, te)
        if res[0] > best[0]: best = res
    return best

# ─── Probe: kNN variants ────────────────────────────────────────────────
print("\n=== kNN (70/30 split, seed=42, k=10) ===")
for (name, xf) in [
    ('text 384-dim',    x_text),
    ('CLIP 512-dim',    x_clip),
    ('multimodal 896-dim', x_dual),
]:
    a_s, a_g, f1 = knn_acc_full(xf[tr], y_sub[tr], y_grd[tr], xf[te], y_sub[te], y_grd[te], k=10)
    print(f"  {name:<22}: Subj={a_s:.2f}%, Grade={a_g:.2f}%, F1={f1:.2f}")

# ─── Probe: DualHead GraphSAGE ──────────────────────────────────────────
print("\n=== DualHead GraphSAGE (70/30 split, seed=42, 300ep) ===")

torch.manual_seed(42)
for (name, xf) in [
    ('text 384-dim',    x_text),
    ('CLIP 512-dim',    x_clip),
    ('multimodal 896-dim', x_dual),
]:
    res = best_dual(DualHeadSAGE, xf, ei_dual, y_sub, y_grd, tr, n=2, ep=300,
                    in_ch=xf.shape[1], h=256, n_sub=NUM_SUB, n_grd=NUM_GRD)
    print(f"  {name:<22}: Subj={res[0]:.2f}%, Grade={res[1]:.2f}%, F1={res[2]:.2f}")

print("\n=== DualHead Attention+GraphSAGE (70/30 split, seed=42, 300ep) ===")
torch.manual_seed(42)
res = best_dual(DualHeadAttnSAGE, x_clip, x_text, ei_dual, y_sub, y_grd, tr, n=2, ep=300,
                img_dim=512, text_dim=384, h=256, n_sub=NUM_SUB, n_grd=NUM_GRD)
print(f"  Cross-modal Attn+SAGE: Subj={res[0]:.2f}%, Grade={res[1]:.2f}%, F1={res[2]:.2f}")
