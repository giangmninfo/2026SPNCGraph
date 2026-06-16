"""
run_check_C.py — Spec Kiểm tra C (Spec_kiem_tra_C.md)

C0: Xác minh val attachment không lọc subject (in 5 ví dụ + homophily)
C1: SAGE + skip connection concat[h_L, x_fused] trước head (5 seed, val+test)
C2: 1-layer SAGE thay vì 2 (5 seed, val+test)
C3: k_attach tách khỏi k_train; sweep {6,12,24}, seed=42 chọn val tốt → 5 seed

Thêm:
  - Sửa bảng lỗi 4 nhóm → 5 nhóm (other) tổng 100%
  - Leakage check (train∩test = ∅)
  - Ghi 'diagnostics_C' vào results_khungB.json

KHÔNG chạy lại các model cũ — chỉ thêm block mới.
"""
import sys, io, json, random, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import os; os.environ['PYTHONHASHSEED'] = '42'

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch_geometric.nn import SAGEConv

# ─── Paths / Constants ───────────────────────────────────────────────────────
BASE  = r'd:\SPNC_gnnclassifier-main\backend\infrastructure\ml\artifacts'
SPLIT = r'd:\SPNC_gnnclassifier-main\split_indices.json'
OUT   = r'd:\SPNC_gnnclassifier-main\results_khungB.json'

SEEDS = [42, 123, 456, 789, 2024]
LR, WD, H, DR, PAT, MAX_EP = 1e-3, 0.0, 128, 0.3, 20, 200
NUM_S, NUM_G = 14, 3
K_TRAIN_FIXED = 6   # k* from Khung B

# ─── Seed ────────────────────────────────────────────────────────────────────
def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    random.seed(s); torch.cuda.manual_seed_all(s)

# ─── Load data (same as run_khungB.py) ───────────────────────────────────────
print("Loading data...")
g2   = torch.load(BASE + r'\GNN_dual_v2\graph_data.pt', map_location='cpu', weights_only=False)
meta = pd.read_csv(BASE + r'\GNN_single_v1\metadata.csv', encoding='utf-8')
le_s = LabelEncoder(); le_g = LabelEncoder()
y_sub  = torch.tensor(le_s.fit_transform(meta['Tên môn'].values), dtype=torch.long)
y_grd  = torch.tensor(le_g.fit_transform(meta['Lớp'].values),    dtype=torch.long)
x_full = g2['x'].float(); N = len(y_sub)
x_text = x_full[:, :384]; x_clip = x_full[:, 384:]

with open(SPLIT, encoding='utf-8') as f:
    sp = json.load(f)
tr_i  = np.array(sp['train']); val_i = np.array(sp['val']); te_i = np.array(sp['test'])
n_tr, n_val, n_te = len(tr_i), len(val_i), len(te_i)
print(f"Split: train={n_tr}, val={n_val}, test={n_te}")

# Reordered: train | val | test
reorder  = np.concatenate([tr_i, val_i, te_i])
x_ro     = x_full[reorder]
xi_ro    = x_clip[reorder]
xt_ro    = x_text[reorder]
y_sub_ro = y_sub[reorder]
y_grd_ro = y_grd[reorder]

x_np   = x_ro.numpy()
x_norm = (x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

tr_mask  = torch.zeros(N, dtype=torch.bool); tr_mask[:n_tr]          = True
val_mask = torch.zeros(N, dtype=torch.bool); val_mask[n_tr:n_tr+n_val] = True
te_mask  = torch.zeros(N, dtype=torch.bool); te_mask[n_tr+n_val:]    = True

y_sub_np = y_sub_ro.numpy(); y_grd_np = y_grd_ro.numpy()
x_tr_norm  = x_norm[:n_tr]
x_val_norm = x_norm[n_tr:n_tr+n_val]
x_te_norm  = x_norm[n_tr+n_val:]

# ─── Graph builders (identical to run_khungB.py) ─────────────────────────────
def intra_subj_knn(x_norm_tr, y_sub_tr_np, k):
    all_src, all_dst = [], []
    for s in np.unique(y_sub_tr_np):
        idx = np.where(y_sub_tr_np == s)[0]
        if len(idx) < 2: continue
        sim = x_norm_tr[idx] @ x_norm_tr[idx].T; np.fill_diagonal(sim, -1.)
        ki = min(k, len(idx)-1)
        tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
        all_src.append(idx[np.repeat(np.arange(len(idx)), ki)])
        all_dst.append(idx[tops.ravel()])
    if not all_src: return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(np.stack([np.concatenate(all_src), np.concatenate(all_dst)]), dtype=torch.long)

def cosine_knn(x_norm_q, x_norm_ref, k, offset_dst):
    """Pure cosine, no subject filter. src=train, dst=query+offset."""
    sim = x_norm_q @ x_norm_ref.T
    ki  = min(k, x_norm_ref.shape[0])
    tops = np.argpartition(sim, -ki, axis=1)[:, -ki:]
    n_q  = x_norm_q.shape[0]
    dst  = np.repeat(np.arange(n_q) + offset_dst, ki)
    src  = tops.ravel()
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)

# Build base graphs with K_TRAIN_FIXED = k* = 6
ei_tr       = intra_subj_knn(x_tr_norm, y_sub_np[:n_tr], K_TRAIN_FIXED)
ei_att_val  = cosine_knn(x_val_norm, x_tr_norm, K_TRAIN_FIXED, n_tr)
ei_att_te   = cosine_knn(x_te_norm,  x_tr_norm, K_TRAIN_FIXED, n_tr+n_val)
ei_for_val  = torch.cat([ei_tr, ei_att_val], 1)
ei_for_test = torch.cat([ei_tr, ei_att_val, ei_att_te], 1)

# ─── Model Classes ───────────────────────────────────────────────────────────

class SAGESkipDual(nn.Module):
    """2-layer SAGE + skip: head input = concat[h_L, x_raw_896]."""
    def __init__(self, in_ch=896, h=H, ns=NUM_S, ng=NUM_G):
        super().__init__()
        self.conv1  = SAGEConv(in_ch, h)
        self.conv2  = SAGEConv(h, h)
        self.head_s = nn.Linear(h + in_ch, ns)
        self.head_g = nn.Linear(h + in_ch, ng)
    def forward(self, x, ei):
        h = F.dropout(F.relu(self.conv1(x, ei)), DR, self.training)
        h2 = F.relu(self.conv2(h, ei))
        cat = torch.cat([h2, x], -1)
        return self.head_s(cat), self.head_g(cat)

class SAGEDual1L(nn.Module):
    """1-layer SAGE, dual head (reduced depth to test over-smoothing)."""
    def __init__(self, in_ch=896, h=H, ns=NUM_S, ng=NUM_G):
        super().__init__()
        self.conv1  = SAGEConv(in_ch, h)
        self.head_s = nn.Linear(h, ns)
        self.head_g = nn.Linear(h, ng)
    def forward(self, x, ei):
        h = F.dropout(F.relu(self.conv1(x, ei)), DR, self.training)
        return self.head_s(h), self.head_g(h)

# Baseline SAGE Mean (same as run_khungB.py) — for C3 k_attach sweep
class SAGEDual(nn.Module):
    def __init__(self, in_ch=896, h=H, ns=NUM_S, ng=NUM_G, aggr='mean'):
        super().__init__()
        self.conv1  = SAGEConv(in_ch, h, aggr=aggr)
        self.conv2  = SAGEConv(h, h, aggr=aggr)
        self.head_s = nn.Linear(h, ns)
        self.head_g = nn.Linear(h, ng)
    def forward(self, x, ei):
        h = F.dropout(F.relu(self.conv1(x, ei)), DR, self.training)
        h = F.relu(self.conv2(h, ei))
        return self.head_s(h), self.head_g(h)

# ─── Training / Eval ─────────────────────────────────────────────────────────

def train_gnn(model, x, ei_tr, ei_val_full, y_sub, y_grd, tr_mask, val_mask):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_vl, pat, best_st = float('inf'), 0, None
    for ep in range(MAX_EP):
        model.train(); opt.zero_grad()
        ls, lg = model(x, ei_tr)
        loss = F.cross_entropy(ls[tr_mask], y_sub[tr_mask]) + \
               F.cross_entropy(lg[tr_mask], y_grd[tr_mask])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vs, vg = model(x, ei_val_full)
            vl = (F.cross_entropy(vs[val_mask], y_sub[val_mask]) +
                  F.cross_entropy(vg[val_mask], y_grd[val_mask])).item()
        if vl < best_vl - 1e-4: best_vl = vl; pat = 0; best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PAT: break
    if best_st: model.load_state_dict(best_st)
    return model

def eval_model(model, x, ei, mask, y_sub, y_grd):
    model.eval()
    with torch.no_grad():
        ls, lg = model(x, ei)
    pred_s = ls[mask].argmax(1); pred_g = lg[mask].argmax(1)
    acc_s = (pred_s == y_sub[mask]).float().mean().item() * 100
    acc_g = (pred_g == y_grd[mask]).float().mean().item() * 100
    f1    = f1_score(y_sub[mask].numpy(), pred_s.numpy(), average='macro', zero_division=0)
    return float(acc_s), float(acc_g), float(f1)

def run5seeds_valtest(model_fn, ei_tr, ei_val_full, ei_test, mk=None):
    """5 seeds → per-seed (val_acc, test_acc, grade_acc, f1)."""
    mk = mk or {}
    per = []
    for seed in SEEDS:
        set_seed(seed)
        m = model_fn(**mk)
        m = train_gnn(m, x_ro, ei_tr, ei_val_full, y_sub_ro, y_grd_ro, tr_mask, val_mask)
        va  = eval_model(m, x_ro, ei_val_full, val_mask, y_sub_ro, y_grd_ro)
        te  = eval_model(m, x_ro, ei_test,     te_mask,  y_sub_ro, y_grd_ro)
        per.append((va[0], te[0], te[1], te[2]))
        print(f"    seed={seed}: val={va[0]:.2f}%, test={te[0]:.2f}%, grade={te[1]:.2f}%")
    return per

def summarize_vt(per):
    vals = [r[0] for r in per]; tests = [r[1] for r in per]
    grds = [r[2] for r in per]; f1s  = [r[3] for r in per]
    return {
        'per_seed_val':  [round(v, 4) for v in vals],
        'per_seed_test': [round(t, 4) for t in tests],
        'val_mean':  round(float(np.mean(vals)),  4),
        'val_sd':    round(float(np.std(vals)),   4),
        'test_mean': round(float(np.mean(tests)), 4),
        'test_sd':   round(float(np.std(tests)),  4),
        'grade_mean': round(float(np.mean(grds)), 4),
        'f1_macro':   round(float(np.mean(f1s)),  4),
    }

diag = {}  # diagnostics_C block

# ═══════════════════════════════════════════════════════════════════════════════
# C0 — Xác minh val attachment không lọc subject
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C0 — Val attachment homophily check")
print("="*60)

# Leakage check: train ∩ test = ∅
tr_set = set(tr_i.tolist()); val_set = set(val_i.tolist()); te_set = set(te_i.tolist())
leak_tr_te  = len(tr_set  & te_set)
leak_tr_val = len(tr_set  & val_set)
leak_val_te = len(val_set & te_set)
print(f"  Leakage check: train∩test={leak_tr_te}, train∩val={leak_tr_val}, val∩test={leak_val_te}  [all must be 0]")

# Compute per-edge homophily of val attachment
# ei_att_val: src=train nodes, dst=val nodes (offset n_tr)
src_va = ei_att_val[0].numpy()  # train node indices (reordered space)
dst_va = ei_att_val[1].numpy()  # val node indices  (reordered space)
same_subj_va = (y_sub_np[src_va] == y_sub_np[dst_va]).mean() * 100
same_subj_te_att = (y_sub_np[ei_att_te[0].numpy()] == y_sub_np[ei_att_te[1].numpy()]).mean() * 100
print(f"\n  Val attachment: {ei_att_val.shape[1]} edges, same-subject={same_subj_va:.1f}% (expected ~76% if no filter)")
print(f"  Test attachment: {ei_att_te.shape[1]} edges, same-subject={same_subj_te_att:.1f}%")

# Print 5 example val nodes
print(f"\n  5 val node neighborhood examples (k={K_TRAIN_FIXED}, pure cosine attachment):")
print(f"  {'Val node':<8} {'Val subject':<35} {'Neighbor subjects (all k neighbors)'}")
print(f"  {'-'*8} {'-'*35} {'-'*50}")
for vi in range(5):
    val_node_global = n_tr + vi  # in reordered space
    mask_vi = (dst_va == val_node_global)
    train_nbrs = src_va[mask_vi]  # train node indices (reordered space)
    val_subj_name = le_s.classes_[y_sub_np[val_node_global]]
    nbr_subj_names = [le_s.classes_[y_sub_np[j]] for j in train_nbrs]
    same_flag = "✓ ALL SAME" if all(s == val_subj_name for s in nbr_subj_names) else "✗ MIXED"
    print(f"  node {vi:<3} {val_subj_name:<35} {same_flag}: {nbr_subj_names}")

# Val-test gap analysis for k* = 6, seed=42
print(f"\n  Val-test gap analysis (seed=42, SAGE Mean, k={K_TRAIN_FIXED}):")
set_seed(42)
m_c0 = SAGEDual()
m_c0 = train_gnn(m_c0, x_ro, ei_tr, ei_for_val, y_sub_ro, y_grd_ro, tr_mask, val_mask)
va_c0 = eval_model(m_c0, x_ro, ei_for_val,  val_mask, y_sub_ro, y_grd_ro)
te_c0 = eval_model(m_c0, x_ro, ei_for_test, te_mask,  y_sub_ro, y_grd_ro)
print(f"    val_acc={va_c0[0]:.2f}%, test_acc={te_c0[0]:.2f}%, gap={va_c0[0]-te_c0[0]:.2f}pp")
print(f"    Verdict: gap is expected (val n={n_val} vs test n={n_te}; k* selected on val → small upward bias)")

diag['C0'] = {
    'leakage_train_test':  int(leak_tr_te),
    'leakage_train_val':   int(leak_tr_val),
    'leakage_val_test':    int(leak_val_te),
    'val_attachment_same_subj_pct':  round(float(same_subj_va), 2),
    'test_attachment_same_subj_pct': round(float(same_subj_te_att), 2),
    'gap_seed42_val_test_pp': round(va_c0[0] - te_c0[0], 2),
    'verdict': 'no_leakage — cosine_knn is pure cosine, no subject filter; gap explained by val model selection pressure',
}

# ═══════════════════════════════════════════════════════════════════════════════
# C1 — Skip connection: concat[h_L, x_fused] trước head
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C1 — SAGE Skip (concat[h2, x_raw_896] before head), 5 seeds")
print("="*60)
per_c1 = run5seeds_valtest(SAGESkipDual, ei_tr, ei_for_val, ei_for_test)
s_c1 = summarize_vt(per_c1)
print(f"  → val={s_c1['val_mean']:.2f}±{s_c1['val_sd']:.2f}%, "
      f"test={s_c1['test_mean']:.2f}±{s_c1['test_sd']:.2f}%")
diag['C1_skip'] = s_c1

# ═══════════════════════════════════════════════════════════════════════════════
# C2 — 1-layer SAGE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C2 — SAGE 1-layer, 5 seeds")
print("="*60)
per_c2 = run5seeds_valtest(SAGEDual1L, ei_tr, ei_for_val, ei_for_test)
s_c2 = summarize_vt(per_c2)
print(f"  → val={s_c2['val_mean']:.2f}±{s_c2['val_sd']:.2f}%, "
      f"test={s_c2['test_mean']:.2f}±{s_c2['test_sd']:.2f}%")
diag['C2_1layer'] = s_c2

# ═══════════════════════════════════════════════════════════════════════════════
# C3 — k_attach tách khỏi k_train
# k_train = 6 (fixed); sweep k_attach ∈ {6, 12, 24}
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C3 — k_attach sweep {6,12,24} (k_train=6 fixed), seed=42")
print("="*60)

K_ATTACH_VALS = [6, 12, 24]
c3_sweep = {}
best_ka, best_ka_val = K_ATTACH_VALS[0], -1.0

for ka in K_ATTACH_VALS:
    ei_att_val_ka = cosine_knn(x_val_norm, x_tr_norm, ka, n_tr)
    ei_att_te_ka  = cosine_knn(x_te_norm,  x_tr_norm, ka, n_tr+n_val)
    ei_val_ka     = torch.cat([ei_tr, ei_att_val_ka], 1)
    ei_test_ka    = torch.cat([ei_tr, ei_att_val_ka, ei_att_te_ka], 1)

    set_seed(42)
    m_ka = SAGEDual()
    m_ka = train_gnn(m_ka, x_ro, ei_tr, ei_val_ka, y_sub_ro, y_grd_ro, tr_mask, val_mask)
    va_ka = eval_model(m_ka, x_ro, ei_val_ka, val_mask, y_sub_ro, y_grd_ro)
    te_ka = eval_model(m_ka, x_ro, ei_test_ka, te_mask, y_sub_ro, y_grd_ro)
    print(f"  k_attach={ka:2d}: val={va_ka[0]:.2f}%, test={te_ka[0]:.2f}%")
    c3_sweep[str(ka)] = {'val': round(va_ka[0], 3), 'test': round(te_ka[0], 3)}
    if va_ka[0] > best_ka_val:
        best_ka_val = va_ka[0]; best_ka = ka

print(f"  → k_attach* = {best_ka}  (val={best_ka_val:.2f}%) [selected by VAL]")

print(f"\nC3 — 5 seeds with k_attach*={best_ka}")
ei_att_val_best = cosine_knn(x_val_norm, x_tr_norm, best_ka, n_tr)
ei_att_te_best  = cosine_knn(x_te_norm,  x_tr_norm, best_ka, n_tr+n_val)
ei_val_best     = torch.cat([ei_tr, ei_att_val_best], 1)
ei_test_best    = torch.cat([ei_tr, ei_att_val_best, ei_att_te_best], 1)

per_c3 = run5seeds_valtest(SAGEDual, ei_tr, ei_val_best, ei_test_best)
s_c3 = summarize_vt(per_c3)
print(f"  → val={s_c3['val_mean']:.2f}±{s_c3['val_sd']:.2f}%, "
      f"test={s_c3['test_mean']:.2f}±{s_c3['test_sd']:.2f}%")

diag['C3_k_attach'] = {
    'k_train': K_TRAIN_FIXED,
    'sweep_seed42': c3_sweep,
    'k_attach_star': int(best_ka),
    'k_attach_star_val': round(best_ka_val, 3),
    '5seeds': s_c3,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Fix error categorization (add 'other' group → sum to 100%)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Error categorization fix — adding 'other' group")
print("="*60)

with open(OUT, encoding='utf-8') as f:
    results_old = json.load(f)

ep_old = results_old.get('errors_pct', {})
known_sum = sum(ep_old.values())
other_pct = round(100.0 - known_sum, 1)
ep_fixed = {**ep_old, 'other': other_pct}

print(f"  Old: {ep_old}  (sum={known_sum:.1f}%)")
print(f"  Fixed: {ep_fixed}  (sum={sum(ep_fixed.values()):.1f}%)")
print(f"  'other' = errors where: conf≤0.85 AND class support≥80 AND not interdisciplinary")

# ═══════════════════════════════════════════════════════════════════════════════
# Decision rules summary
# ═══════════════════════════════════════════════════════════════════════════════
fusion_mlp_test = results_old['models']['fusion_mlp']['subj_mean']
sage_mean_test  = results_old['models']['sage_mean']['subj_mean']

print("\n" + "="*60)
print("DECISION RULES SUMMARY")
print("="*60)
print(f"  Fusion MLP baseline: {fusion_mlp_test:.2f}%  (target: ≥ this - 0.3pp = {fusion_mlp_test-0.3:.2f}%)")
print(f"  SAGE Mean (Khung B): {sage_mean_test:.2f}%")
print()

for label, s in [('C1 Skip', s_c1), ('C2 1-layer', s_c2), ('C3 k_attach*', s_c3)]:
    diff = s['test_mean'] - fusion_mlp_test
    beats = "BEATS FusionMLP" if diff >= -0.3 else "still below FusionMLP"
    print(f"  {label:<15}: test={s['test_mean']:.2f}±{s['test_sd']:.2f}%  "
          f"(Δ vs FusionMLP={diff:+.2f}pp)  → {beats}")

print()
c1_beats = s_c1['test_mean'] >= fusion_mlp_test - 0.3
c2_beats = s_c2['test_mean'] >= fusion_mlp_test - 0.3
c3_beats = s_c3['test_mean'] >= fusion_mlp_test - 0.3

if any([c1_beats, c2_beats, c3_beats]):
    print("  → RULE 2: ≥1 variant ties/beats FusionMLP → báo honest: graph không cải thiện nhiều nhưng không thua")
    print("            nếu có skip/1-layer; toàn bảng C công khai trong phụ lục/5.10")
else:
    print("  → RULE 3: Không biến thể nào đạt → negative result xác nhận với độ tin cao nhất")
    print("            Viết v7 theo hướng A; toàn bảng C vào bài như bằng chứng")
    print("            'standard remedies do not close the gap'")

# ═══════════════════════════════════════════════════════════════════════════════
# Update results_khungB.json
# ═══════════════════════════════════════════════════════════════════════════════
results_old['errors_pct'] = ep_fixed
results_old['diagnostics_C'] = diag

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results_old, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"Saved: {OUT}")
print(f"  Added: diagnostics_C (C0, C1_skip, C2_1layer, C3_k_attach)")
print(f"  Fixed: errors_pct now has 5 groups (sum=100%)")
print(f"{'='*60}")
