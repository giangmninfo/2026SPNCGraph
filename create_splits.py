"""
Create and freeze train/val/test indices for release.

Split strategy:
  - Step 1: 70/30 stratified on 42-class composite (subject x grade), seed=42
            → 6750 train, 2894 pool
  - Step 2: pool split 1/3 val / 2/3 test, stratified by 14-class subject, seed=42
            → 965 val, 1929 test

Output: data_splits.npz  (indices + metadata, frozen for release)
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd, torch, json
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub  = le_s.fit_transform(meta['Tên môn'].values)   # 0..13
y_grd  = le_g.fit_transform(meta['Lớp'].values)       # 0..2
y_comp = y_sub * 3 + y_grd                             # 0..41
N = len(y_comp)

print(f"N = {N}, subjects = {len(le_s.classes_)}, grades = {len(le_g.classes_)}")

# ── Step 1: 70/30 outer split ─────────────────────────────────────────────────
spl1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr_i, pool_i = next(spl1.split(np.arange(N), y_comp))
assert len(tr_i) == 6750, f"Expected 6750 train, got {len(tr_i)}"
assert len(pool_i) == 2894, f"Expected 2894 pool, got {len(pool_i)}"

# ── Step 2: pool → val / test (stratified by subject, seed=42) ───────────────
# target: 965 val (≈1/3), 1929 test (≈2/3)
target_test_frac = 1929 / len(pool_i)   # ≈ 0.6665
spl2 = StratifiedShuffleSplit(n_splits=1, test_size=target_test_frac, random_state=42)
val_local, test_local = next(spl2.split(np.arange(len(pool_i)), y_sub[pool_i]))
val_i  = pool_i[val_local]
test_i = pool_i[test_local]

n_tr = len(tr_i); n_val = len(val_i); n_te = len(test_i)
assert n_tr + n_val + n_te == N, f"Index leak: {n_tr}+{n_val}+{n_te} ≠ {N}"
assert len(set(tr_i) & set(val_i)) == 0, "train/val overlap"
assert len(set(tr_i) & set(test_i)) == 0, "train/test overlap"
assert len(set(val_i) & set(test_i)) == 0, "val/test overlap"

print(f"\nSplit sizes:  train={n_tr}  val={n_val}  test={n_te}")

# ── Per-subject distribution check ───────────────────────────────────────────
print(f"\n{'Subject':<45} {'Total':>6} {'Train%':>7} {'Val%':>6} {'Test%':>6}")
print("-" * 72)
for s, name in enumerate(le_s.classes_):
    total = (y_sub == s).sum()
    t_cnt = (y_sub[tr_i] == s).sum()
    v_cnt = (y_sub[val_i] == s).sum()
    te_cnt = (y_sub[test_i] == s).sum()
    print(f"  {name:<43} {total:>6}  {t_cnt/total*100:>6.1f}% {v_cnt/total*100:>5.1f}% {te_cnt/total*100:>5.1f}%")
print(f"  {'TOTAL':<43} {N:>6}  {n_tr/N*100:>6.1f}% {n_val/N*100:>5.1f}% {n_te/N*100:>5.1f}%")

# ── Per-grade distribution ────────────────────────────────────────────────────
print(f"\n{'Grade':<10} {'Total':>6} {'Train%':>7} {'Val%':>6} {'Test%':>6}")
for g, name in enumerate(le_g.classes_):
    total = (y_grd == g).sum()
    t_cnt = (y_grd[tr_i] == g).sum()
    v_cnt = (y_grd[val_i] == g).sum()
    te_cnt = (y_grd[test_i] == g).sum()
    print(f"  {str(name):<8} {total:>6}  {t_cnt/total*100:>6.1f}% {v_cnt/total*100:>5.1f}% {te_cnt/total*100:>5.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = r'd:\SPNC_gnnclassifier-main\data_splits.npz'
np.savez(
    out_path,
    train=np.sort(tr_i).astype(np.int32),
    val=np.sort(val_i).astype(np.int32),
    test=np.sort(test_i).astype(np.int32),
    # scalar metadata stored as 0-d arrays
    N_total=np.array(N, dtype=np.int32),
    seed_outer=np.array(42, dtype=np.int32),
    seed_inner=np.array(42, dtype=np.int32),
    subject_names=np.array(le_s.classes_),
    grade_names=np.array([str(g) for g in le_g.classes_]),
)
print(f"\nSaved: {out_path}")

# Quick reload check
d = np.load(out_path, allow_pickle=True)
assert len(d['train']) == n_tr
assert len(d['val'])   == n_val
assert len(d['test'])  == n_te
print(f"Reload OK: train={len(d['train'])}, val={len(d['val'])}, test={len(d['test'])}")

# ── Print loading snippet for README ─────────────────────────────────────────
print("""
--- Usage snippet ---
import numpy as np
d = np.load('data_splits.npz', allow_pickle=True)
train_idx = d['train']   # (6750,) int32
val_idx   = d['val']     # (965,)  int32
test_idx  = d['test']    # (1929,) int32
""")
