# backend/infrastructure/ml/classifiers/voting_classifier.py

import torch
import numpy as np
from typing import Dict


class VotingClassifier:
    """
    Similarity-based voting classifier.

    Consumes:
        - node features (image + text fused)

    Produces:
        - raw vote scores for subjects, grades, and subject-grade pairs

    This classifier is:
        - Model-agnostic
        - Graph-free
        - Dimension-agnostic (depends on injected feature DB)
    """

    def __init__(
        self,
        features_db: np.ndarray,   # shape: (N, D)
        metadata_df,               # must contain subject & grade columns
        top_k: int = 10,
        subject_penalties: Dict[str, float] | None = None,
    ):
        self.features_db = features_db
        self.df = metadata_df
        self.top_k = top_k

        self.subject_penalties = subject_penalties or {
            "Mĩ thuật": 0.85,
            "Hoạt động trải nghiệm, hướng nghiệp": 0.85,
        }

    def classify(self, node_feat: torch.Tensor) -> dict:
        """
        Args:
            node_feat: Tensor shape (1, D)

        Returns:
            {
                "subjects": Dict[str, float],
                "grades": Dict[str, float],
                "pairs": Dict[str, float]
            }
        """
        query = node_feat.cpu().numpy()  # (1, D)

        # --- cosine similarity ---
        db_norm = self.features_db / np.linalg.norm(
            self.features_db, axis=1, keepdims=True
        )
        q_norm = query / np.linalg.norm(query, axis=1, keepdims=True)

        scores = (db_norm @ q_norm.T).squeeze(1)  # (N,)

        # --- top-k neighbors ---
        top_indices = np.argsort(scores)[-self.top_k:][::-1]

        vote_subject: Dict[str, float] = {}
        vote_grade: Dict[str, float] = {}
        pair_scores: Dict[str, float] = {}

        for idx in top_indices:
            row = self.df.iloc[idx]

            subject = str(row["Tên môn"])
            grade = str(row["Lớp"]).replace(".0", "")
            label = f"{subject} - {grade}"
            score = float(scores[idx])

            penalty = self.subject_penalties.get(subject, 1.0)

            vote_subject[subject] = vote_subject.get(subject, 0.0) + score * penalty
            vote_grade[grade] = vote_grade.get(grade, 0.0) + score
            pair_scores[label] = pair_scores.get(label, 0.0) + score

        return {
            "subjects": vote_subject,
            "grades": vote_grade,
            "pairs": pair_scores,
        }
