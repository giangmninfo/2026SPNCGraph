import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch, pandas as pd, numpy as np

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'

# Check node_embeddings
emb = torch.load(BASE + r'\GNN_dual_v2\node_embeddings.pt', map_location='cpu', weights_only=False)
print("=== node_embeddings.pt ===")
print("Type:", type(emb))
if isinstance(emb, dict):
    for k, v in emb.items():
        print(f"  {k}: {type(v).__name__} {getattr(v,'shape','')}")
else:
    print(emb)

# Check graph_data dual
print("\n=== GNN_dual_v2/graph_data.pt ===")
g2 = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
print("Type:", type(g2))
if isinstance(g2, dict):
    for k, v in g2.items():
        sh = getattr(v, 'shape', '')
        unique = ''
        if hasattr(v, 'shape') and len(v.shape) == 1 and v.shape[0] < 10000:
            uniq = torch.unique(v)
            if uniq.shape[0] < 50:
                unique = f'unique={uniq.tolist()}'
        print(f"  {k}: {type(v).__name__} {sh}  {unique}")

# Check graphsage_subject.pt architecture
import json
with open(BASE + r'\GNN_dual_v2\subject_labels.json', encoding='utf-8') as f:
    subj_labels = json.load(f)
print("\n=== Subject labels in dual_v2 ===")
print(subj_labels)

# Load trained model to check architecture
from torch_geometric.nn import SAGEConv
import torch.nn as nn

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, ei):
        import torch.nn.functional as F
        x = F.relu(self.conv1(x, ei))
        return self.conv2(x, ei)

model = GraphSAGE(896, 256, 14)
sd = torch.load(BASE + r'\GNN_dual_v2\graphsage_subject.pt', map_location='cpu')
print("\n=== graphsage_subject.pt state dict keys ===")
for k, v in sd.items():
    print(f"  {k}: {v.shape}")

# Evaluate the pre-trained model on the full graph
g1 = torch.load(BASE + r'\GNN_single_v1\graph_data_multimodal.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_subject = torch.tensor(le.fit_transform(meta['Tên môn'].values), dtype=torch.long)
print("\n=== Class mapping ===")
for i, c in enumerate(le.classes_):
    print(f"  {i}: {c}")

# evaluate pretrained graphsage_subject on ENTIRE graph
x_dual = g2['x'].float()
ei_dual = g2['edge_index'].long()

model.load_state_dict(sd)
model.eval()
with torch.no_grad():
    out = model(x_dual, ei_dual)
    pred = out.argmax(dim=1)  # 0-13

print("\n=== Pretrained subject model predictions ===")
print("pred unique:", torch.unique(pred).tolist())
print("pred distribution:", {int(v): int((pred==v).sum()) for v in torch.unique(pred)})

# The pred values: need to match against y_subject
# But labels in graphsage_subject might use different subject_labels mapping
# Check: subject_labels.json in dual_v2
print("\nChecking if subject_label matches graph node order...")
# In dual_v2, each node label should match subject_labels.json (not LabelEncoder order)
# Let's check if g2 has y labels
print("g2 keys:", list(g2.keys()))

# Maybe the graph has train/test masks?
