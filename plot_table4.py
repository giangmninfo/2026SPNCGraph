import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data (từ results_khungB.json) ────────────────────────────────────────────
labels = [
    'Strategy 1\n(Intra-subj\nkNN, k=6)',
    'Feature-only\nkNN (k=6)',
    'Random\nEdge Ctrl',
    'Threshold\n(τ=0.75)',
    'Semi-\nTransductive\n(upper bound)',
    'Metadata\n(upper bound)',
]
means = [87.8072, 90.0052, 90.1711, 91.9544, 97.4287, 98.4448]
sds   = [0.1778,  0.3135,  0.4430,  0.1107,  0.1067,  0.1270]

# Category: proposed / no-label / label-informed
# 0: Strategy 1 (proposed, highlighted)
# 1,2,3: alternative graph (no label)
# 4,5: upper bound (label-informed)
BLUE_DARK  = '#1f77b4'
BLUE_MID   = '#4e9ac7'
BLUE_LIGHT = '#aec7e8'
ORANGE     = '#ff7f0e'
GRAY_UB    = '#9e9e9e'

colors = [BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_LIGHT, GRAY_UB, GRAY_UB]
hatches = ['', '', '', '', '///', '///']

fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(labels))
width = 0.58

bars = ax.bar(x, means, width=width, color=colors, edgecolor='white',
              linewidth=0.8, zorder=3,
              yerr=sds, capsize=4,
              error_kw=dict(elinewidth=1.2, ecolor='#333', capthick=1.2, zorder=5))

# Apply hatches for upper bounds
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)
    bar.set_edgecolor('#666' if h else 'white')

# Value labels
for bar, m, s in zip(bars, means, sds):
    ypos = bar.get_height() + s + 0.3
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{m:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Divider between no-label and upper bound
ax.axvline(3.5, color='#aaa', lw=1.0, ls='--', alpha=0.7, zorder=2)
ax.text(3.6, 96.8, 'Label-informed\n(upper bound)', fontsize=8,
        color='#666', va='top')

# Highlight Strategy 1 border
bars[0].set_edgecolor(BLUE_DARK)
bars[0].set_linewidth(2.0)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel('Subject Accuracy (%)', fontsize=11)
ax.set_ylim(82, 102)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_alpha(0.4)
ax.spines['bottom'].set_alpha(0.4)
ax.tick_params(axis='x', length=0)

# Legend
legend_handles = [
    mpatches.Patch(facecolor=BLUE_DARK,  edgecolor='white', label='Strategy 1 (proposed)'),
    mpatches.Patch(facecolor=BLUE_MID,   edgecolor='white', label='Alternative graph (no label)'),
    mpatches.Patch(facecolor=GRAY_UB,    edgecolor='#666',  hatch='///', label='Upper bound (label-informed)'),
]
ax.legend(handles=legend_handles, fontsize=8.5, framealpha=0.7,
          loc='upper left', bbox_to_anchor=(0.01, 0.99))

plt.tight_layout()
OUT = r'd:\SPNC_gnnclassifier-main\barchart_table4.png'
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
