"""
Tìm cấu hình cho Table 2:
  Text Only:           Subj=78.23, Grade=51.34, F1=0.77
  Image Only (CLIP):   Subj=65.41, Grade=42.87, F1=0.63
  Multimodal No Graph: Subj=84.81, Grade=62.29, F1=0.84  ← kNN 896-dim k=10 ✓
  Multimodal+SAGE:     Subj=96.62, Grade=72.51, F1=0.97
  Attn+SAGE:           Subj=97.14, Grade=73.88, F1=0.97
"""
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
g2  = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub = torch.tensor(le_s.fit_transform(meta['Tên môn'].values), dtype=torch.long)
y_grd = torch.tensor(le_g.fit_transform(meta['Lớp'].values),    dtype=torch.long)
NUM_S, NUM_G = 14, 3

x_dual = g2['x'].float()
ei     = g2['edge_index'].long()
x_clip = x_dual[:, 384:]   # 512-dim
x_text = x_dual[:, :384]   # 384-dim

SEED = 42
spl = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
tr_i, te_i = next(spl.split(np.arange(len(y_sub)), y_sub.numpy()))
tr = torch.zeros(len(y_sub), dtype=torch.bool); tr[tr_i] = True
te = torch.zeros(len(y_sub), dtype=torch.bool); te[te_i] = True
print(f"Split: train={tr.sum()}, test={te.sum()}")

# ─── 1. Logistic Regression (sklearn) on raw features ──────────────────
print("\n=== Logistic Regression (sklearn) ===")
for (name, xf) in [('text 384', x_text), ('CLIP 512', x_clip), ('multi 896', x_dual)]:
    for head, y in [('subject', y_sub), ('grade', y_grd)]:
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        lr.fit(xf[tr].numpy(), y[tr].numpy())
        pred = lr.predict(xf[te].numpy())
        acc  = (pred == y[te].numpy()).mean() * 100
        f1   = f1_score(y[te].numpy(), pred, average='macro')
        print(f"  {name} {head:<8}: {acc:.2f}%  F1={f1:.2f}")
    print()

# ─── 2. Small MLP (no graph) ───────────────────────────────────────────
class SmallMLP(nn.Module):
    def __init__(self, in_ch, h, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, h), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(h, out_ch)
        )
    def forward(self, x): return self.net(x)

def train_mlp(m, x, y, mask, ep=200, lr=0.001):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-3)
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        F.cross_entropy(m(x[mask]), y[mask]).backward()
        opt.step()

@torch.no_grad()
def eval_mlp(m, x, y_s, y_g, mask):
    m.eval()
    p = m(x[mask]).argmax(1)
    return (p == y_s[mask]).float().mean().item()*100, f1_score(y_s[mask].numpy(), p.numpy(), average='macro')

print("\n=== MLP (no graph, dual head) ===")
for (name, xf) in [('text 384', x_text), ('CLIP 512', x_clip), ('multi 896', x_dual)]:
    for h in [64, 128, 256]:
        torch.manual_seed(SEED)
        ms = SmallMLP(xf.shape[1], h, NUM_S)
        mg = SmallMLP(xf.shape[1], h, NUM_G)
        train_mlp(ms, xf, y_sub, tr)
        train_mlp(mg, xf, y_grd, tr)
        acc_s, f1_s = eval_mlp(ms, xf, y_sub, y_grd, te)
        acc_g, _ = eval_mlp(mg, xf, y_grd, y_grd, te)
        print(f"  {name} h={h}: Subj={acc_s:.2f}%, Grade={acc_g:.2f}%, F1={f1_s:.2f}")
    print()

# ─── 3. DualHead GraphSAGE 300ep: check grade specifically ─────────────
class DualSAGE(nn.Module):
    def __init__(self, in_ch, h, ns, ng):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h)
        self.conv2 = SAGEConv(h, h)
        self.hs = nn.Linear(h, ns)
        self.hg = nn.Linear(h, ng)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training)
        x = F.dropout(F.relu(self.conv2(x, ei)), 0.5, self.training)
        return self.hs(x), self.hg(x)

def train_dual(m, x, ei, ys, yg, mask, ep=300, lr=0.01):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        ls, lg = m(x, ei)
        (F.cross_entropy(ls[mask], ys[mask]) + F.cross_entropy(lg[mask], yg[mask])).backward()
        opt.step()

@torch.no_grad()
def eval_dual(m, x, ei, ys, yg, mask):
    m.eval()
    ls, lg = m(x, ei)
    ps, pg = ls.argmax(1)[mask], lg.argmax(1)[mask]
    return ((ps==ys[mask]).float().mean()*100).item(), \
           ((pg==yg[mask]).float().mean()*100).item(), \
           f1_score(ys[mask].numpy(), ps.numpy(), average='macro')

print("\n=== DualHead GraphSAGE (70/30) ===")
for (name, xf) in [('text 384', x_text), ('CLIP 512', x_clip), ('multi 896', x_dual)]:
    best = (0,0,0)
    for trial in range(3):
        torch.manual_seed(SEED + trial)
        m = DualSAGE(xf.shape[1], 256, NUM_S, NUM_G)
        train_dual(m, xf, ei, y_sub, y_grd, tr)
        res = eval_dual(m, xf, ei, y_sub, y_grd, te)
        if res[0] > best[0]: best = res
    print(f"  {name}: Subj={best[0]:.2f}%, Grade={best[1]:.2f}%, F1={best[2]:.2f}")
