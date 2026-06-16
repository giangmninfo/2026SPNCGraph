import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np, json, pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE  = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
SPLIT = r'd:\SPNC_gnnclassifier-main\split_indices.json'
TSNE  = r'd:\SPNC_gnnclassifier-main\tsne_coords.npy'
OUT   = r'd:\SPNC_gnnclassifier-main\tsne_visualization.png'

# ── Load ─────────────────────────────────────────────────────────────────────
coords = np.load(TSNE)                        # (1929, 2)

meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')
le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub_all = le_s.fit_transform(meta['Tên môn'].values)
y_grd_all = le_g.fit_transform(meta['Lớp'].values)

with open(SPLIT, encoding='utf-8') as f:
    sp = json.load(f)
te_i = np.array(sp['test'])

y_sub = y_sub_all[te_i]   # 1929 subject labels (0–13)
y_grd = y_grd_all[te_i]   # 1929 grade labels (0–2)

# Short subject names
subj_names_full = le_s.classes_
subj_short = {
    'Âm nhạc': 'Music',
    'Công nghệ': 'Tech.',
    'Giáo dục quốc phòng và an ninh': 'Defence',
    'Hoạt động trải nghiệm, hướng nghiệp': 'Activity',
    'Hóa học': 'Chem.',
    'Lịch sử': 'History',
    'Mĩ thuật': 'Fine Art',
    'Ngữ Văn': 'Lit.',
    'Sinh học': 'Biology',
    'Tiếng Anh': 'English',
    'Tin học': 'CS',
    'Toán': 'Math',
    'Vật Lý': 'Physics',
    'Địa lí': 'Geog.',
}
subj_labels = [subj_short.get(n, n) for n in subj_names_full]
grade_labels = [f'Grade {le_g.classes_[i]}' for i in range(3)]

# ── Color palettes ────────────────────────────────────────────────────────────
# 14 distinct colors for subjects
SUBJ_COLORS = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#999999','#66c2a5','#fc8d62',
    '#8da0cb','#e78ac3','#a6d854','#ffd92f',
]
# 3 colors for grades — warm palette to show overlap clearly
GRADE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

DOT = 6   # marker size

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(wspace=0.08)

# ── (a) Subject — clear separation ───────────────────────────────────────────
for i, (name, color) in enumerate(zip(subj_labels, SUBJ_COLORS)):
    mask = y_sub == i
    ax1.scatter(coords[mask, 0], coords[mask, 1],
                c=color, s=DOT, alpha=0.7, linewidths=0,
                label=name, zorder=3)

# Cluster centroid labels
for i, (name, color) in enumerate(zip(subj_labels, SUBJ_COLORS)):
    mask = y_sub == i
    if mask.sum() == 0: continue
    cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
    ax1.text(cx, cy, name, fontsize=6.5, ha='center', va='center',
             color='white', fontweight='bold',
             path_effects=[pe.withStroke(linewidth=2.2, foreground=color)])

ax1.set_title('(a) Subject embeddings\n(14 classes — clear separation)',
              fontsize=11, pad=8)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.spines[:].set_visible(False)

leg1 = ax1.legend(fontsize=7, ncol=2, framealpha=0.7,
                  loc='lower left', markerscale=2.0,
                  handletextpad=0.4, columnspacing=0.8)

# ── (b) Grade — significant overlap ──────────────────────────────────────────
grade_order = [2, 1, 0]   # draw lower grades last so visible
for i in grade_order:
    mask = y_grd == i
    ax2.scatter(coords[mask, 0], coords[mask, 1],
                c=GRADE_COLORS[i], s=DOT, alpha=0.45, linewidths=0,
                label=grade_labels[i], zorder=3)

ax2.set_title('(b) Grade embeddings\n(3 classes — significant overlap)',
              fontsize=11, pad=8)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.spines[:].set_visible(False)

leg2 = ax2.legend(fontsize=9, framealpha=0.75,
                  loc='lower left', markerscale=2.5,
                  handletextpad=0.5)

# Shared caption note
fig.text(0.5, 0.01,
         't-SNE of GNN embeddings (GraphSAGE Mean, seed 42, test set, n = 1 929)',
         ha='center', fontsize=8.5, color='#555')

plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
print(f'  coords shape: {coords.shape}')
print(f'  subjects: {len(np.unique(y_sub))} classes')
print(f'  grades  : {le_g.classes_.tolist()}')
