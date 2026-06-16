import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import pandas as pd
import json

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'

# GNN_dual_v2
print('=== GNN_dual_v2/graph_data.pt ===')
g2 = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
if isinstance(g2, dict):
    for k, v in g2.items():
        sh = getattr(v, 'shape', '')
        dt = getattr(v, 'dtype', '')
        print(f'  {k}: {type(v).__name__} {sh} {dt}')
else:
    print(type(g2))
    for attr in ['x', 'edge_index', 'y', 'num_nodes', 'train_mask', 'test_mask', 'val_mask']:
        val = getattr(g2, attr, None)
        if val is not None:
            sh = getattr(val, 'shape', val)
            print(f'  {attr}: {sh}')

print()

# GNN_single_v1 multimodal
print('=== GNN_single_v1/graph_data_multimodal.pt ===')
g1 = torch.load(BASE + r'\GNN_single_v1\graph_data_multimodal.pt', map_location='cpu', weights_only=False)
for attr in ['x', 'edge_index', 'y', 'num_nodes', 'train_mask', 'test_mask', 'val_mask']:
    val = getattr(g1, attr, None)
    if val is not None:
        sh = getattr(val, 'shape', val)
        unique = torch.unique(val).tolist() if hasattr(val, 'shape') and len(val.shape) == 1 and val.shape[0] < 200 else ''
        print(f'  {attr}: {sh}  unique={unique}')

print()
# Check y values
print('y unique values:', torch.unique(g1.y).tolist())
print('y value counts:', {int(v): int((g1.y == v).sum()) for v in torch.unique(g1.y)})

# metadata
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')
print('\n  metadata label values:', sorted(meta['label'].unique()))
print('  metadata Ten mon values:', sorted(meta['Tên môn'].unique()))

# Map subject to int
subjects = sorted(meta['Tên môn'].unique())
subject2idx = {s: i for i, s in enumerate(subjects)}
meta['subject_label'] = meta['Tên môn'].map(subject2idx)
print('\n  subject_label distribution:')
print(meta['subject_label'].value_counts().sort_index())

# Verify order: metadata rows match graph nodes
print('\n  First 5 composite labels from metadata:')
for i in range(5):
    row = meta.iloc[i]
    print(f"  [{i}] {row['Tên môn']} lop {row['Lớp']} => graph y={g1.y[i].item()}")
