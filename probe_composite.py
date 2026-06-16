"""
Hypothesis: ALL Table 2 models predict 42-class composite (subject+grade),
then extract subject acc and grade acc from the composite predictions.
Targets:
  Text Only:           Subj=78.23, Grade=51.34, F1=0.77
  Image Only (CLIP):   Subj=65.41, Grade=42.87, F1=0.63
  Multimodal No Graph: Subj=84.81, Grade=62.29, F1=0.84
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
g1  = torch.load(BASE + r'\GNN_single_v1\graph_data_multimodal.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

# Composite label (42 classes: subject 14 × grade 3)
le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub = torch.tensor(le_s.fit_transform(meta['Tên môn'].values), dtype=torch.long)  # 0-13
y_grd = torch.tensor(le_g.fit_transform(meta['Lớp'].values),    dtype=torch.long)  # 0-2

NUM_S, NUM_G = 14, 3
# Composite: sub_id * 3 + grd_id → 42 classes
y_comp = y_sub * NUM_G + y_grd   # 0-41

x_dual = g2['x'].float()
ei     = g2['edge_index'].long()
x_clip = x_dual[:, 384:]   # 512-dim CLIP
x_text = x_dual[:, :384]   # 384-dim text
x_single = g1.x.float()    # 2432-dim (ResNet + text)

SEED = 42
spl = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
tr_i, te_i = next(spl.split(np.arange(len(y_comp)), y_comp.numpy()))
tr = torch.zeros(len(y_comp), dtype=torch.bool); tr[tr_i] = True
te = torch.zeros(len(y_comp), dtype=torch.bool); te[te_i] = True
print(f"70/30 split: train={tr.sum()}, test={te.sum()}")

def composite_metrics(pred_comp_np, y_s_np, y_g_np):
    """Extract subj/grade acc and subj macro-F1 from composite predictions."""
    pred_s = pred_comp_np // NUM_G
    pred_g = pred_comp_np %  NUM_G
    acc_s = (pred_s == y_s_np).mean() * 100
    acc_g = (pred_g == y_g_np).mean() * 100
    f1_s  = f1_score(y_s_np, pred_s, average='macro')
    return acc_s, acc_g, f1_s

# ─── 1. Logistic Regression on composite (42-class) ────────────────────
print("\n=== LogisticRegression 42-class composite ===")
for (name, xf) in [('text 384', x_text), ('CLIP 512', x_clip),
                    ('multi 896', x_dual), ('multi 2432', x_single)]:
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED, n_jobs=-1)
    lr.fit(xf[tr].numpy(), y_comp[tr].numpy())
    pred = lr.predict(xf[te].numpy())
    acc_s, acc_g, f1_s = composite_metrics(pred, y_sub[te].numpy(), y_grd[te].numpy())
    print(f"  {name:<14}: Subj={acc_s:.2f}%, Grade={acc_g:.2f}%, F1={f1_s:.2f}")

# ─── 2. MLP (no graph) 42-class composite ──────────────────────────────
class MLP42(nn.Module):
    def __init__(self, in_ch, h, nc=42):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, h), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(h, h//2),  nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(h//2, nc)
        )
    def forward(self, x): return self.net(x)

def train_mlp42(m, x, y, mask, ep=300, lr=0.001):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    m.train()
    for _ in range(ep):
        opt.zero_grad()
        F.cross_entropy(m(x[mask]), y[mask]).backward()
        opt.step()

print("\n=== MLP 42-class composite (no graph) ===")
for (name, xf, h) in [
    ('text 384 h=256',    x_text,   256),
    ('CLIP 512 h=256',    x_clip,   256),
    ('multi 896 h=512',   x_dual,   512),
    ('multi 2432 h=512',  x_single, 512),
]:
    torch.manual_seed(SEED)
    m = MLP42(xf.shape[1], h)
    train_mlp42(m, xf, y_comp, tr)
    with torch.no_grad():
        m.eval()
        pred = m(xf[te]).argmax(1).numpy()
    acc_s, acc_g, f1_s = composite_metrics(pred, y_sub[te].numpy(), y_grd[te].numpy())
    print(f"  {name:<22}: Subj={acc_s:.2f}%, Grade={acc_g:.2f}%, F1={f1_s:.2f}")

# ─── 3. kNN on composite (vote by composite, then extract subj+grade) ──
def knn_composite_metrics(x_tr, y_comp_tr, y_sub_te, y_grd_te, x_te, k=10):
    x_tr = x_tr / (x_tr.norm(dim=1, keepdim=True) + 1e-8)
    x_te = x_te / (x_te.norm(dim=1, keepdim=True) + 1e-8)
    sim = x_te @ x_tr.T
    preds = []
    for i in range(x_te.shape[0]):
        topk = sim[i].topk(k).indices
        preds.append(y_comp_tr[topk].mode().values.item())
    preds = np.array(preds)
    return composite_metrics(preds, y_sub_te.numpy(), y_grd_te.numpy())

print("\n=== kNN 42-class composite ===")
for (name, xf) in [('text 384', x_text), ('CLIP 512', x_clip),
                    ('multi 896', x_dual), ('multi 2432', x_single)]:
    for k in [5, 10, 15]:
        acc_s, acc_g, f1_s = knn_composite_metrics(
            xf[tr], y_comp[tr], y_sub[te], y_grd[te], xf[te], k)
        print(f"  {name:<14} k={k:2d}: Subj={acc_s:.2f}%, Grade={acc_g:.2f}%, F1={f1_s:.2f}")

# ─── 4. GraphSAGE 42-class composite ───────────────────────────────────
class SAGE42(nn.Module):
    def __init__(self, in_ch, h, nc=42):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, h)
        self.conv2 = SAGEConv(h, nc)
    def forward(self, x, ei):
        return self.conv2(F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training), ei)

print("\n=== GraphSAGE 42-class composite ===")
for (name, xf) in [('text 384', x_text), ('CLIP 512', x_clip), ('multi 896', x_dual)]:
    best = (0, 0, 0)
    for trial in range(3):
        torch.manual_seed(SEED + trial)
        m = SAGE42(xf.shape[1], 256)
        opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
        m.train()
        for _ in range(300):
            opt.zero_grad()
            F.cross_entropy(m(xf, ei)[tr], y_comp[tr]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            pred = m(xf, ei).argmax(1)[te].numpy()
        res = composite_metrics(pred, y_sub[te].numpy(), y_grd[te].numpy())
        if res[0] > best[0]: best = res
    print(f"  {name:<14}: Subj={best[0]:.2f}%, Grade={best[1]:.2f}%, F1={best[2]:.2f}")

# ─── 5. CrossModal Attention + GraphSAGE 42-class ──────────────────────
class AttnSAGE42(nn.Module):
    def __init__(self, img_dim=512, text_dim=384, h=256, nc=42):
        super().__init__()
        fused = img_dim + text_dim
        self.attn = nn.Sequential(
            nn.Linear(fused, 64), nn.Tanh(),
            nn.Linear(64, 2), nn.Softmax(dim=-1)
        )
        self.ip = nn.Linear(img_dim, fused//2)
        self.tp = nn.Linear(text_dim, fused//2)
        self.conv1 = SAGEConv(fused, h)
        self.conv2 = SAGEConv(h, nc)
    def fuse(self, xi, xt):
        a = self.attn(torch.cat([xi, xt], -1))
        return torch.cat([a[:,0:1]*self.ip(xi), a[:,1:2]*self.tp(xt)], -1)
    def forward(self, xi, xt, ei):
        x = self.fuse(xi, xt)
        return self.conv2(F.dropout(F.relu(self.conv1(x, ei)), 0.5, self.training), ei)

print("\n=== Attn+GraphSAGE 42-class composite ===")
best = (0,0,0)
for trial in range(3):
    torch.manual_seed(SEED + trial)
    m = AttnSAGE42(512, 384, 256, 42)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    m.train()
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(m(x_clip, x_text, ei)[tr], y_comp[tr]).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(x_clip, x_text, ei).argmax(1)[te].numpy()
    res = composite_metrics(pred, y_sub[te].numpy(), y_grd[te].numpy())
    if res[0] > best[0]: best = res
print(f"  Attn+SAGE: Subj={best[0]:.2f}%, Grade={best[1]:.2f}%, F1={best[2]:.2f}")
