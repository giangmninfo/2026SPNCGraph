import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json

OUT = r'd:\SPNC_gnnclassifier-main\confusion_matrix.png'

with open(r'd:\SPNC_gnnclassifier-main\results_khungB.json', encoding='utf-8') as f:
    R = json.load(f)

cm = np.array(R['confusion_matrix'])   # (14, 14)

# Subject names — order from table1_per_class keys = le_s.classes_ order
labels = [
    'Technology', 'Nat. Defence', 'Exp. Activities',
    'Chemistry', 'History', 'Fine Arts',
    'Literature', 'Biology', 'Informatics',
    'English', 'Mathematics', 'Physics',
    'Music', 'Geography',
]

# Normalize by row (true class) → recall per cell
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8.5))

im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label('Recall (row-normalized)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Cell annotations — show count if ≥ 1, color text by background
n = len(labels)
thresh = 0.5
for i in range(n):
    for j in range(n):
        val_n  = cm[i, j]
        val_pct = cm_norm[i, j]
        if val_n == 0:
            continue
        color = 'white' if val_pct > thresh else '#222'
        # Diagonal: show % + count; off-diagonal: count only if > 0
        if i == j:
            txt = f'{val_pct*100:.0f}%\n({val_n})'
            fs  = 7.5
        else:
            txt = str(int(val_n))
            fs  = 7
        ax.text(j, i, txt, ha='center', va='center',
                fontsize=fs, color=color, fontweight='bold' if i == j else 'normal')

# Axes
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8.5)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel('Predicted label', fontsize=10, labelpad=8)
ax.set_ylabel('True label', fontsize=10, labelpad=8)
ax.set_title('Confusion Matrix — Subject Classification\n'
             '(GraphSAGE Mean, seed 42, test set n = 1 929)',
             fontsize=11, pad=10)

# Grid lines between cells
for i in range(n + 1):
    ax.axhline(i - 0.5, color='white', lw=0.5)
    ax.axvline(i - 0.5, color='white', lw=0.5)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')

# Top off-diagonal errors
print('\nTop off-diagonal errors:')
cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
flat = [(int(cm_off[i,j]), labels[i], labels[j])
        for i in range(n) for j in range(n) if cm_off[i,j] > 0]
for cnt, true, pred in sorted(flat, reverse=True)[:8]:
    print(f'  {true} → {pred}: {cnt}')
