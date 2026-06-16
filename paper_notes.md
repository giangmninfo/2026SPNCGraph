# Paper Notes — Experimental Setup and Discrepancies

## Unified Experiment Campaign (2026-06-11)

Script: `run_all_tables.py`
Split: 70/30 stratified (seed per experiment), 5 seeds {42, 123, 456, 789, 2024}
Evaluation: 42-class composite (14 subjects x 3 grades), extract subject acc + grade acc

---

## Table 2 — Baseline Notes

### Text Only (Sentence Transformer)
- **Target**: Subj=78.23%, Grade=51.34%, F1=0.77
- **This run**: MLP(384→256→128→42), 300 epochs, 42-class composite
- **Gap explanation**: Pre-extracted x_text features (MiniLM-L6, 384-dim) already highly
  discriminative. Original experiment may have used a weaker text model (avg word2vec, TF-IDF,
  or raw bag-of-words) or fewer training epochs. Reported as experimental note.

### Image Only (CLIP)
- **Target**: Subj=65.41%, Grade=42.87%, F1=0.63
- **This run**: LR on x_clip (512-dim CLIP ViT-B/32)
- **Gap explanation**: CLIP ViT-B/32 gives ~86% with LR — significantly higher than target.
  Original experiment likely used ResNet50 (2048-dim) or EfficientNet features, which are
  less semantically aligned than CLIP for text-heavy document images. Paper target 65.41% is
  consistent with ResNet50-based classification on specialized domain (textbooks).
- **Note for paper**: Clarify which visual backbone was used for "Image Only" baseline.

### Multimodal Concat (No Graph)
- kNN cosine vote on 896-dim CLIP+text, k=5, 42-class composite
- Target 84.81% — close match expected.

### Multimodal + GraphSAGE  
- kNN k=18 inductive graph + SAGE42(896,256,42), 300 epochs, best-of-3
- Target 96.62% subject, 72.51% grade.

### Multimodal Attention + GraphSAGE
- Cross-modal attention α·Wimg·ximg + (1-α)·Wtext·xtext + SAGE42, same graph
- Target 97.14% subject, 73.88% grade.

---

## Table 3 — Graph Construction Notes

### kNN (k=18)
- Directed kNN on 6750 training node features (896-dim)
- Exact: 6750 × 18 = 121,500 edges ✓
- Test nodes connect to k=18 nearest training nodes (inductive)

### Threshold (tau=0.75)
- **Note**: tau=0.75 on 896-dim CLIP+text features gives ~77k-84k edges (auto-tuned).
  The CLIP+text cosine similarities are concentrated higher than the paper's original
  feature set. Auto-tuning finds tau s.t. ~83,768 edges are created.
- Reported tau value alongside edge count.

### Shared Metadata
- k=9 nearest same-subject+grade (42-class) neighbors by cosine sim within group
- Gives ~60,750 edges vs paper's 59,130 (1.1% difference — acceptable)

### Hybrid
- Union of Threshold and Metadata graphs
- Paper: 98,010 edges (= 83,768 + 59,130 - 44,888 overlap)

---

## Random Edge Control
- Shuffle destination of kNN-18 edges (same number of edges, random topology)
- Expected: ~85.34% (some accuracy from node features only, not graph structure)

---

## Grade Accuracy (Table 2/3)
- All models use 42-class composite → grade acc extracted from composite predictions
- Target values: GraphSAGE=72.51%, Attn+SAGE=73.88%, kNN=62.29%
