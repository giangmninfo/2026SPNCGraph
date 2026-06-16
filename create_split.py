"""
Tạo và đóng băng train/val/test indices cho release (Khung B spec).

Step 1: Outer 70/30 split, stratify on 42-class composite, seed=42 → 6750 train, 2894 pool
Step 2: Pool → val (965) + test (1929), stratify by 14-class subject, seed=42
        dùng train_test_split (như spec) thay vì StratifiedShuffleSplit

Output: split_indices.json {train:[...], val:[...], test:[...]}
Mọi script sau đều load từ file này — KHÔNG tự split lại.
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd, json, torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

BASE = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')

le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub  = le_s.fit_transform(meta['Tên môn'].values)  # 0..13
y_grd  = le_g.fit_transform(meta['Lớp'].values)      # 0..2
y_comp = (y_sub * 3 + y_grd).astype(int)
N = len(y_comp)
print(f"N={N}, n_subjects={len(le_s.classes_)}, n_grades={len(le_g.classes_)}")

# ── Step 1: outer 70/30 split ────────────────────────────────────────────────
spl = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr_i, pool_i = next(spl.split(np.arange(N), y_comp))
assert len(tr_i) == 6750 and len(pool_i) == 2894, f"{len(tr_i)}, {len(pool_i)}"

# ── Step 2: pool → val (965) + test (1929) ───────────────────────────────────
# Chính xác theo spec: train_test_split với test_size=1929, train_size=965
val_i, test_i = train_test_split(
    pool_i,
    test_size=1929, train_size=965,
    stratify=y_sub[pool_i],
    random_state=42
)
val_i  = np.asarray(val_i)
test_i = np.asarray(test_i)

print(f"train={len(tr_i)}, val={len(val_i)}, test={len(test_i)}")

# ── Integrity checks ─────────────────────────────────────────────────────────
assert len(tr_i) + len(val_i) + len(test_i) == N
assert len(set(tr_i) & set(val_i))  == 0, "train/val overlap"
assert len(set(tr_i) & set(test_i)) == 0, "train/test overlap"
assert len(set(val_i) & set(test_i)) == 0, "val/test overlap"

# ── Per-subject distribution ──────────────────────────────────────────────────
print(f"\n{'Subject':<45} {'Tot':>5} {'Tr%':>6} {'Va%':>6} {'Te%':>6}")
print("-" * 72)
for s, name in enumerate(le_s.classes_):
    tot = int((y_sub == s).sum())
    t_n = int((y_sub[tr_i] == s).sum())
    v_n = int((y_sub[val_i] == s).sum())
    e_n = int((y_sub[test_i] == s).sum())
    print(f"  {name:<43} {tot:>5}  {t_n/tot*100:>4.1f}%  {v_n/tot*100:>4.1f}%  {e_n/tot*100:>4.1f}%")
print(f"  {'TOTAL':<43} {N:>5}  {len(tr_i)/N*100:>4.1f}%  {len(val_i)/N*100:>4.1f}%  {len(test_i)/N*100:>4.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
split = {
    "train": sorted(int(x) for x in tr_i),
    "val":   sorted(int(x) for x in val_i),
    "test":  sorted(int(x) for x in test_i),
    "meta": {
        "N_total": int(N),
        "n_train": int(len(tr_i)), "n_val": int(len(val_i)), "n_test": int(len(test_i)),
        "seed_outer": 42, "seed_inner": 42,
        "stratify_outer": "42-class composite (subject x grade)",
        "stratify_inner": "14-class subject",
        "method_outer": "StratifiedShuffleSplit test_size=0.3",
        "method_inner": "train_test_split test_size=1929 train_size=965",
        "subject_names": le_s.classes_.tolist(),
        "grade_names": [str(g) for g in le_g.classes_],
    }
}

out = r'd:\SPNC_gnnclassifier-main\split_indices.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(split, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out}")

# Quick reload check
with open(out, encoding='utf-8') as f:
    d = json.load(f)
assert d['meta']['n_train'] == 6750 and d['meta']['n_val'] == 965 and d['meta']['n_test'] == 1929
print("Reload OK ✓")
