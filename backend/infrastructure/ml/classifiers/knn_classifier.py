import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNClassifier:
    def __init__(self, embeddings, labels, k=5):
        self.labels = labels
        self.knn = NearestNeighbors(
            n_neighbors=k,
            metric="cosine"
        )
        self.knn.fit(embeddings)

    def predict_scores(self, z, topk=5):
        dists, idxs = self.knn.kneighbors(z, n_neighbors=topk)

        scores = {}
        for i, d in zip(idxs[0], dists[0]):
            w = 1.0 / (d + 1e-6)
            lbl = self.labels[i].item()
            scores[lbl] = scores.get(lbl, 0) + w

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class GraphSAGEKNNClassifier:
    def __init__(
        self,
        encoder,
        subject_knn,
        grade_knn,
        subject_labels,
        grade_labels,
        temperature=4.0,
        alpha=0.7,
        beta=0.3,
    ):
        self.encoder = encoder
        self.subject_knn = subject_knn
        self.grade_knn = grade_knn
        self.subject_labels = subject_labels
        self.grade_labels = grade_labels
        self.T = temperature
        self.alpha = alpha
        self.beta = beta

    def _to_confidence(self, results):
        labels, scores = zip(*results)
        scores = np.array(scores, dtype=np.float32)

        exp = np.exp((scores - scores.max()) / self.T)
        probs = exp / exp.sum()

        return list(zip(labels, probs))

    def classify(self, x, topk=5):
        z = self.encoder.encode(x).cpu().numpy()

        subj_raw = self.subject_knn.predict_scores(z)
        grade_raw = self.grade_knn.predict_scores(z)

        subj = self._to_confidence([
            (self.subject_labels[str(i)], s) for i, s in subj_raw
        ])

        grade = self._to_confidence([
            (self.grade_labels[str(i)], s) for i, s in grade_raw
        ])

        # 🔑 RETURN DUAL-HEAD OUTPUT
        return (
            {label: float(score) for label, score in subj[:topk]},
            {label: float(score) for label, score in grade[:topk]},
        )
