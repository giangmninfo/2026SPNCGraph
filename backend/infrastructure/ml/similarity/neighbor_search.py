# infrastructure/ml/similarity/neighbor_search.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class NeighborSearcher:
    def __init__(self, x: np.ndarray):
        self.x = x

    def find(self, new_feat: np.ndarray, k: int):
        sims = cosine_similarity(new_feat, self.x)[0]
        return sims.argsort()[-k:]
