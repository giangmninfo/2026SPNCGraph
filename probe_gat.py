import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
g2  = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')
le  = LabelEncoder()
y_sub = torch.tensor(le.fit_transform(meta['Tên môn'].values), dtype=torch.long)
NUM_CLASSES = 14

x_dual = g2['x'].float()
ei_dual = g2['edge_index'].long()
x_img_clip = x_dual[:, 384:]   # 512-dim CLIP only

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(np.arange(len(y_sub)), y_sub.numpy()))
train_mask = torch.zeros(len(y_sub), dtype=torch.bool); train_mask[train_idx] = True
test_mask  = torch.zeros(len(y_sub), dtype=torch.bool); test_mask[test_idx]  = True

class GATModel(nn.Module):
    def __init__(self, in_ch, h, out_ch, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_ch, h, heads=heads, dropout=0.3)
        self.conv2 = GATConv(h*heads, out_ch, heads=1, concat=False, dropout=0.3)
    def forward(self, x, ei):
        x = F.dropout(F.elu(self.conv1(x, ei)), p=0.5, training=self.training)
        return self.conv2(x, ei)

# Track epoch-by-epoch accuracy for GAT with 896-dim and 512-dim
print("GAT 896-dim: accuracy over epochs")
torch.manual_seed(42)
m_full = GATModel(896, 64, NUM_CLASSES, heads=4)
opt = torch.optim.Adam(m_full.parameters(), lr=0.01, weight_decay=5e-4)

for ep in range(1, 401):
    m_full.train()
    opt.zero_grad()
    F.cross_entropy(m_full(x_dual, ei_dual)[train_mask], y_sub[train_mask]).backward()
    opt.step()
    if ep in [30, 50, 80, 100, 120, 150, 200, 300, 400]:
        m_full.eval()
        with torch.no_grad():
            acc = (m_full(x_dual, ei_dual).argmax(1)[test_mask] == y_sub[test_mask]).float().mean().item()*100
        print(f"  epoch {ep:3d}: {acc:.2f}%")

print("\nGAT 512-dim: accuracy over epochs")
torch.manual_seed(42)
m_clip = GATModel(512, 64, NUM_CLASSES, heads=4)
opt2 = torch.optim.Adam(m_clip.parameters(), lr=0.01, weight_decay=5e-4)

for ep in range(1, 601):
    m_clip.train()
    opt2.zero_grad()
    F.cross_entropy(m_clip(x_img_clip, ei_dual)[train_mask], y_sub[train_mask]).backward()
    opt2.step()
    if ep in [50, 100, 150, 200, 300, 400, 500, 600]:
        m_clip.eval()
        with torch.no_grad():
            acc = (m_clip(x_img_clip, ei_dual).argmax(1)[test_mask] == y_sub[test_mask]).float().mean().item()*100
        print(f"  epoch {ep:3d}: {acc:.2f}%")
