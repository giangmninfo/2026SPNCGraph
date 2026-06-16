# backend/infrastructure/ml/classifiers/knn_graphsage_classifier.py

import torch
import torch.nn.functional as F


class KNNGraphSAGEClassifier:
    """
    Inductive classifier:
    - kNN neighbor retrieval (cosine similarity)
    - Local ego-graph construction
    - GraphSAGE inference
    """

    def __init__(
        self,
        assets,
        neighbor_searcher,
        k_neighbors: int = 10,
        topk: int = 3,
    ):
        self.assets = assets
        self.searcher = neighbor_searcher
        self.k = k_neighbors
        self.topk = topk

    def classify(self, node_feat: torch.Tensor):
        """
        Args:
            node_feat: Tensor (1, D)

        Returns:
            List[(label, probability)]
        """

        # ---- 1. kNN neighbor retrieval (Voting-style) ----
        neighbors = self.searcher.find(
            node_feat.numpy(),
            self.k
        )

        # ---- 2. Build local ego graph ----
        x_local = torch.cat(
            [self.assets.x[neighbors], node_feat],
            dim=0
        )  # (k+1, D)

        new_node_id = x_local.size(0) - 1

        edges = []
        for i in range(len(neighbors)):
            edges.append([i, new_node_id])
            edges.append([new_node_id, i])

        edge_index = torch.tensor(edges).t().long()

        # ---- 3. GraphSAGE inference ----
        with torch.no_grad():
            logits = self.assets.model(x_local, edge_index)[new_node_id]
            probs = F.softmax(logits, dim=0)

        # ---- 4. Decode composite labels ----
        top_probs, top_ids = torch.topk(probs, self.topk)

        return [
            (self.assets.inv_label_map[int(i)], float(p))
            for p, i in zip(top_probs, top_ids)
        ]

class GraphSAGEKNNImageClassifier:
    def __init__(
        self,
        node_feature_builder,
        knn_classifier
    ):
        self.node_feature_builder = node_feature_builder
        self.knn_classifier = knn_classifier

    def classify(self, image_path, topk=3):
        x = self.node_feature_builder.build(image_path)
        return self.knn_classifier.predict(x, topk)
