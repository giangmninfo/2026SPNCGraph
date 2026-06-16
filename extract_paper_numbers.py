"""
extract_paper_numbers.py
Trích toàn bộ số liệu cần cho bài báo từ results_khungB.json + tính thêm:
  - 95% CI từng model (t_{0.025,4}=2.776)
  - Params Fusion MLP
  - Graph stats upper bounds (edges, mean_deg)
  - [Hg] % attachment neighbors khác grade
  - Image feature duplicate check across splits
  - κs/κg/BLEU → N/A, ghi rõ
In ra màn hình; không ghi thêm file.
"""
import sys, io, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, torch
from sklearn.preprocessing import LabelEncoder
from scipy import stats as sp_stats

# ── Load ────────────────────────────────────────────────────────────────────
BASE  = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
SPLIT = r'd:\SPNC_gnnclassifier-main\split_indices.json'
OUT   = r'd:\SPNC_gnnclassifier-main\results_khungB.json'

with open(OUT, encoding='utf-8') as f:
    R = json.load(f)

g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')
le_g = LabelEncoder()
y_grd = le_g.fit_transform(meta['Lớp'].values)
x_full = g2['x'].float().numpy()

with open(SPLIT, encoding='utf-8') as f:
    sp = json.load(f)
tr_i  = np.array(sp['train']); val_i = np.array(sp['val']); te_i = np.array(sp['test'])
n_tr, n_val, n_te = len(tr_i), len(val_i), len(te_i)

reorder = np.concatenate([tr_i, val_i, te_i])
x_ro    = x_full[reorder]
y_grd_ro = y_grd[reorder]
x_norm  = (x_ro / (np.linalg.norm(x_ro, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

T95_4 = 2.7764   # t_{0.025, df=4}

def ci95(vals):
    a = np.array(vals)
    m, s = a.mean(), a.std(ddof=0)   # population std (같은 공식으로 SD 계산)
    se = s / np.sqrt(len(a))
    return m, s, m - T95_4*se, m + T95_4*se

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TABLE 2 — Per-seed (5 giá trị) + 95% CI")
print("=" * 70)

models_order = [
    ('fusion_mlp',   'Fusion MLP'),
    ('sage_mean',    'GraphSAGE Mean'),
    ('attn_sage',    'AttnSAGE'),
    ('gcn',          'GCN'),
    ('gat',          'GAT'),
    ('sage_pool',    'SAGE Pool'),
    ('sage_lstm',    'SAGE LSTM'),
    ('knn_voting',   'kNN-Voting'),
    ('text_only',    'Text-only MLP'),
    ('image_only',   'Image-only (CLIP+LR)'),
]

for key, name in models_order:
    m = R['models'][key]
    seeds = m['per_seed_subj']
    mn, sd, lo, hi = ci95(seeds)
    print(f"\n{name}")
    print(f"  Per-seed : {[round(v,4) for v in seeds]}")
    print(f"  Mean±SD  : {mn:.4f} ± {sd:.4f}")
    print(f"  95% CI   : [{lo:.4f}, {hi:.4f}]")
    print(f"  Grade    : {m['grade_mean']:.4f}")
    print(f"  Prec/Rec/F1 (macro): {m['prec_macro']:.4f} / {m['rec_macro']:.4f} / {m['f1_macro']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 3 — Pairwise stats (ours=sage_mean vs each model)")
print("=" * 70)
print(f"{'Model':<20} {'Δpp':>7} {'95%CI':>22} {'t':>8} {'p':>10} {'p_bon':>10} {'d':>8}")
print("-" * 90)
for key, label in [
    ('attn_sage',  'AttnSAGE'),
    ('gcn',        'GCN'),
    ('gat',        'GAT'),
    ('fusion_mlp', 'FusionMLP'),
    ('knn_voting', 'kNN-Voting'),
    ('text_only',  'Text-only'),
    ('image_only', 'Image-only'),
]:
    s = R['stats_vs_ours'][key]
    ci = s['ci95']
    print(f"{label:<20} {s['delta_pp']:>+7.3f} [{ci[0]:>7.3f},{ci[1]:>7.3f}] "
          f"{s['t']:>8.3f} {s['p']:>10.4f} {s['p_bonferroni']:>10.4f} {s['d']:>+8.3f}")

# [pA] sage_mean vs attn_sage
s_a = R['stats_vs_ours']['attn_sage']
print(f"\n[pA] sage_mean vs attn_sage: p={s_a['p']:.4f}, p_Bonferroni={s_a['p_bonferroni']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FUSION MLP — Parameter count")
print("=" * 70)
in_ch, h, ns, ng = 896, 128, 14, 3
p1 = in_ch*(h*2) + (h*2)          # Linear(896, 256)
p2 = (h*2)*h    + h                # Linear(256, 128)
ph_s = h*ns + ns                   # Linear(128, 14)
ph_g = h*ng + ng                   # Linear(128, 3)
total = p1 + p2 + ph_s + ph_g
print(f"  Linear(896→256):  {p1:>8,}")
print(f"  Linear(256→128):  {p2:>8,}")
print(f"  Head_s(128→14):   {ph_s:>8,}")
print(f"  Head_g(128→3):    {ph_g:>8,}")
print(f"  TOTAL:            {total:>8,}  params")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 1 — Per-class P/R/F1 (sage_mean, seed=42)")
print("=" * 70)
print(f"  {'Môn':<45} P      R      F1     N")
print(f"  {'-'*45} {'----':<7}{'----':<7}{'----':<7}{'----'}")
for name, v in R['table1_per_class'].items():
    print(f"  {name:<45} {v['prec']:.3f}  {v['rec']:.3f}  {v['f1']:.3f}  {v['n']}")
m1 = R['table1_macro']
print(f"  {'Macro':<45} {m1['prec']:.3f}  {m1['rec']:.3f}  {m1['f1']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 5 — Graph statistics")
print("=" * 70)

def intra_subj_knn(x_norm_tr, y_sub_tr, k):
    import numpy as np
    all_src, all_dst = [], []
    for s in np.unique(y_sub_tr):
        idx = np.where(y_sub_tr == s)[0]
        if len(idx) < 2: continue
        sim = x_norm_tr[idx] @ x_norm_tr[idx].T; np.fill_diagonal(sim, -1.)
        ki = min(k, len(idx)-1)
        tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
        all_src.append(idx[np.repeat(np.arange(len(idx)), ki)])
        all_dst.append(idx[tops.ravel()])
    if not all_src: return np.zeros((2,0), dtype=int)
    return np.stack([np.concatenate(all_src), np.concatenate(all_dst)])

# Load subject labels
le_s = LabelEncoder()
y_sub = le_s.fit_transform(meta['Tên môn'].values)
y_sub_ro = y_sub[reorder]
y_grd_ro2 = y_grd[reorder]

def graph_stats_np(src, dst, n_ref, y_sub, y_grd, label):
    n_edges = len(src)
    mean_deg = n_edges / n_ref if n_ref > 0 else 0
    ss = (y_sub[src] == y_sub[dst]).mean()*100 if n_edges else 0
    sg = (y_grd[src] == y_grd[dst]).mean()*100 if n_edges else 0
    print(f"  {label:<25}: edges={n_edges:>7,}, mean_deg={mean_deg:>7.2f}, "
          f"same_subj={ss:.1f}%, same_grade={sg:.1f}%")
    return n_edges, mean_deg, ss, sg

# Strategy 1 (k=6 intra-subj on train)
g1 = R['graphs']['strategy1']
print(f"\n  Strategy 1 (k=6 intra-subj train):")
print(f"    edges={g1['edges']:,}, mean_deg={g1['mean_deg']:.2f}, "
      f"same_subj={g1['same_subj_pct']:.1f}%, same_grade={g1['same_grade_pct']:.1f}%")
print(f"    [Hs] same-grade train edges = {g1['same_grade_pct']:.1f}%")
print(f"    [Hg2] attachment same-subject = {g1['attach_subject_homophily_pct']:.1f}%")
print(f"    val attach edges={g1['attach_edges_val']}, test attach edges={g1['attach_edges_test']}")

# Feature-only (k=6 cosine on train)
gf = R['graphs']['feature_only']
print(f"\n  Feature-only kNN (k=6):")
print(f"    edges={gf['edges']:,}, mean_deg={gf['mean_deg']:.2f}, "
      f"same_subj={gf['same_subj_pct']:.1f}%, same_grade={gf['same_grade_pct']:.1f}%")

# Threshold (τ=0.75)
gt = R['graphs']['threshold']
print(f"\n  Threshold (τ={gt['tau']}):")
print(f"    edges={gt['edges']:,}, mean_deg={gt['mean_deg']:.2f}, "
      f"same_subj={gt['same_subj_pct']:.1f}%, same_grade={gt['same_grade_pct']:.1f}%")
print(f"    [dτ] mean_deg = {gt['mean_deg']:.2f}")

# Semi-transductive and Metadata — rebuild to get stats
print(f"\n  Rebuilding upper bound graphs to get stats...")
x_all_norm = (x_full / (np.linalg.norm(x_full, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
y_sub_all = y_sub  # original order
y_grd_all = y_grd
N_all = len(y_sub_all)

# Semi-transductive: intra-subject kNN on ALL 9644 nodes with k=k*
k_star = R['k_star']  # 6
ei_semi = intra_subj_knn(x_all_norm, y_sub_all, k_star)
src_s, dst_s = ei_semi[0], ei_semi[1]
graph_stats_np(src_s, dst_s, N_all, y_sub_all, y_grd_all, "Semi-transductive")

# Metadata: intra-class42 kNN on ALL nodes (subject × grade)
y_comp42 = y_sub_all * 3 + y_grd_all  # composite 42-class
all_src_m, all_dst_m = [], []
for c in np.unique(y_comp42):
    idx = np.where(y_comp42 == c)[0]
    if len(idx) < 2: continue
    sim = x_all_norm[idx] @ x_all_norm[idx].T; np.fill_diagonal(sim, -1.)
    ki = min(k_star, len(idx)-1)
    tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
    all_src_m.append(idx[np.repeat(np.arange(len(idx)), ki)])
    all_dst_m.append(idx[tops.ravel()])
src_m = np.concatenate(all_src_m); dst_m = np.concatenate(all_dst_m)
graph_stats_np(src_m, dst_m, N_all, y_sub_all, y_grd_all, "Metadata (composite42)")

# [Hg] — % attachment edges (test→train) with DIFFERENT grade
x_te_norm = x_all_norm[te_i]
x_tr_norm = x_all_norm[tr_i]
k = k_star
sim_te_tr = x_te_norm @ x_tr_norm.T
tops = np.argpartition(sim_te_tr, -k, axis=1)[:, -k:]
src_att = tops.ravel()                      # indices into tr_i
dst_att = np.repeat(np.arange(n_te), k)    # test node local index
g_src = y_grd_all[tr_i[src_att]]
g_dst = y_grd_all[te_i[dst_att]]
hg_diff = (g_src != g_dst).mean() * 100
hg_same = 100 - hg_diff
print(f"\n  [Hg] Attachment (test→train) same-grade={hg_same:.1f}%, diff-grade={hg_diff:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("k-SWEEP — 5 val acc values (seed=42, SAGE Mean, Strategy 1)")
print("=" * 70)
for k_str, va in R['k_sweep'].items():
    print(f"  k={k_str:>2}: val_acc={va:.3f}%")
print(f"  → k* = {R['k_star']}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MỤC 4.1 — Annotation quality / leakage")
print("=" * 70)

# [zero] Feature-level duplicate check across splits (cosine sim ≈ 1.0)
x_tr_f = x_norm[:n_tr]
x_val_f = x_norm[n_tr:n_tr+n_val]
x_te_f  = x_norm[n_tr+n_val:]

# Check val vs train
sim_val_tr = x_val_f @ x_tr_f.T  # (n_val, n_tr)
max_sim_val = sim_val_tr.max(axis=1)
dup_val = (max_sim_val > 0.9999).sum()

# Check test vs train
BATCH = 500
max_sim_te = np.zeros(n_te, dtype=np.float32)
for i0 in range(0, n_te, BATCH):
    blk = x_te_f[i0:i0+BATCH] @ x_tr_f.T
    max_sim_te[i0:i0+BATCH] = blk.max(axis=1)
dup_te = (max_sim_te > 0.9999).sum()

# Check val vs test
sim_val_te = x_val_f @ x_te_f.T
max_sim_val_te = sim_val_te.max(axis=1)
dup_val_te = (max_sim_val_te > 0.9999).sum()

print(f"\n  Feature-duplicate check (cosine sim > 0.9999 = near-identical embeddings):")
print(f"    val nodes with duplicate in train : {dup_val}")
print(f"    test nodes with duplicate in train: {dup_te}")
print(f"    val nodes with duplicate in test  : {dup_val_te}")
if dup_val == 0 and dup_te == 0 and dup_val_te == 0:
    print(f"    → [zero] NO near-duplicate content found across splits ✓")
else:
    print(f"    → WARNING: near-duplicates detected — review these nodes")
    if dup_te > 0:
        dup_idx = np.where(max_sim_te > 0.9999)[0]
        print(f"    Duplicate test indices (reordered space): {dup_idx[:10].tolist()}")

print(f"\n  [κs], [κg]: Không đo được từ graph_data.pt")
print(f"    → Cần mẫu gán nhãn kép (double-annotated subset) từ annotation process")
print(f"    → Nếu không có: XÓA câu 'κs=X, κg=Y' trong mục 4.1; thay bằng")
print(f"       'Inter-annotator agreement was not formally measured; labels were")
print(f"        assigned by domain experts following a fixed subject-grade taxonomy.'")

print(f"\n  [B] BLEU-2: Không đo được từ features (đã embed sẵn thành 384-dim)")
print(f"    → XÓA câu đề cập BLEU-2 trong bài; chỉ báo cáo những gì đo được")

print(f"\nDONE.")
