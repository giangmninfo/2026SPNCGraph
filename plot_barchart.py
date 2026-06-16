import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

models = [
    'Fusion MLP',
    'GraphSAGE\nMean',
    'AttnSAGE',
    'kNN-Voting',
    'GAT',
    'GCN',
]
accs = [92.97, 87.81, 86.79, 84.50, 84.42, 83.78]
sds  = [0.12,  0.18,  0.24,  0.00,  0.12,  0.27]

# Colors: highlight Fusion MLP and GraphSAGE
colors = ['#2ca02c', '#1f77b4', '#4e9ac7', '#aec7e8', '#9ecde8', '#c5dcee']

fig, ax = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(models))
bars = ax.bar(x, accs, width=0.55, color=colors, edgecolor='white',
              linewidth=0.8, zorder=3,
              yerr=sds, capsize=4,
              error_kw=dict(elinewidth=1.2, ecolor='#444', capthick=1.2, zorder=4))

# Value labels on top of bars
for bar, acc, sd in zip(bars, accs, sds):
    ypos = bar.get_height() + sd + 0.25
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{acc:.2f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('Subject Accuracy (%)', fontsize=11)
ax.set_ylim(78, 97)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
ax.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_alpha(0.4)
ax.spines['bottom'].set_alpha(0.4)
ax.tick_params(axis='x', length=0)

# Dashed line at Fusion MLP level
ax.axhline(92.97, color='#2ca02c', lw=1.0, ls=':', alpha=0.6, zorder=2)

plt.tight_layout()
OUT = r'd:\SPNC_gnnclassifier-main\barchart_models.png'
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
