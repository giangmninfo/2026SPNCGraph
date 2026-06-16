import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
configs = [
    'Text only\n(Sentence\nTransformer)',
    'Image only\n(CLIP)',
    'Multimodal\nconcat\n(no graph)',
    'Multimodal\nconcat\n+ GraphSAGE',
    'Multimodal\nattention\n+ GraphSAGE',
]
subj_acc = [67.27, 85.74, 92.97, 87.81, 86.79]
subj_sd  = [0.33,  0.00,  0.12,  0.18,  0.24]
grade_acc= [53.67, 55.57, 69.33, 66.13, 62.51]
f1_score = [58.0,  80.1,  90.6,  82.7,  81.0]   # ×100

n = len(configs)
x = np.arange(n)
w = 0.23   # bar width

C_SUBJ  = '#1f77b4'
C_GRADE = '#ff7f0e'
C_F1    = '#2ca02c'
EDGE_BEST = '#d62728'

fig, ax = plt.subplots(figsize=(11, 5.2))

b1 = ax.bar(x - w, subj_acc,  w, label='Subj. Acc (%)',  color=C_SUBJ,  zorder=3,
            yerr=subj_sd, capsize=3,
            error_kw=dict(elinewidth=1.1, ecolor='#333', capthick=1.1, zorder=5))
b2 = ax.bar(x,      grade_acc, w, label='Grade Acc (%)', color=C_GRADE, zorder=3)
b3 = ax.bar(x + w,  f1_score,  w, label='F1 × 100',     color=C_F1,    zorder=3)

# Subj Acc value labels (top of bar)
for i, (bar, v, sd) in enumerate(zip(b1, subj_acc, subj_sd)):
    ypos = v + sd + 0.6
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{v:.2f}', ha='center', va='bottom', fontsize=7.8,
            fontweight='bold', color=C_SUBJ)

# Grade Acc value labels
for bar, v in zip(b2, grade_acc):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.6,
            f'{v:.2f}', ha='center', va='bottom', fontsize=7.8, color=C_GRADE)

# F1 value labels
for bar, v in zip(b3, f1_score):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.6,
            f'{v/100:.3f}', ha='center', va='bottom', fontsize=7.8, color=C_F1)

# Highlight best config (idx=2: Multimodal concat no graph)
for bar in [b1[2], b2[2], b3[2]]:
    bar.set_edgecolor(EDGE_BEST)
    bar.set_linewidth(1.8)

# Background shading by group
ax.axvspan(-0.6,  0.5, alpha=0.04, color='gray', zorder=0)   # unimodal
ax.axvspan( 0.5,  1.5, alpha=0.04, color='gray', zorder=0)
ax.axvspan( 1.5,  2.5, alpha=0.08, color='green', zorder=0)  # best
ax.axvspan( 2.5,  4.6, alpha=0.04, color='blue', zorder=0)   # + graph

# Bracket annotations
ax.annotate('', xy=(0.5, 96), xytext=(-0.5, 96),
            arrowprops=dict(arrowstyle='-', color='#888', lw=1.0))
ax.text(0.0, 96.8, 'Unimodal', ha='center', fontsize=8, color='#555')

ax.annotate('', xy=(4.5, 96), xytext=(2.5, 96),
            arrowprops=dict(arrowstyle='-', color='#888', lw=1.0))
ax.text(3.5, 96.8, '+ Graph', ha='center', fontsize=8, color='#555')

ax.text(2.0, 96.8, '★ Best', ha='center', fontsize=8.5,
        color=EDGE_BEST, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_ylim(40, 100)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_alpha(0.4)
ax.spines['bottom'].set_alpha(0.4)
ax.tick_params(axis='x', length=0)

ax.legend(fontsize=9, framealpha=0.7, loc='lower right')

plt.tight_layout()
OUT = r'd:\SPNC_gnnclassifier-main\ablation_table4.png'
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
