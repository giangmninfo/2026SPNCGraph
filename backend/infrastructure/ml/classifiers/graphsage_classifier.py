# infrastructure/ml/classifiers/graphsage_classifier.py
import torch
import torch.nn.functional as F

class GraphSAGEClassifier:
    def __init__(self, assets, neighbor_searcher):
        self.assets = assets
        self.searcher = neighbor_searcher

    def predict_topk(
        self,
        node_feat: torch.Tensor,
        model,
        labels,
        k_neighbors=5,
        topk=3
    ):
        neighbors = self.searcher.find(
            node_feat.numpy(),
            k_neighbors
        )

        new_node_id = self.assets.x.size(0)
        x_all = torch.cat([self.assets.x, node_feat], dim=0)

        edges = []
        for n in neighbors:
            edges += [[n, new_node_id], [new_node_id, n]]

        edge_index_new = torch.cat(
            [self.assets.edge_index, torch.tensor(edges).t().long()],
            dim=1
        )

        with torch.no_grad():
            logits = model(x_all, edge_index_new)[new_node_id]
            probs = F.softmax(logits, dim=0)

        top_probs, top_ids = torch.topk(probs, topk)
        return [
            (labels[str(i.item())], float(p))
            for p, i in zip(top_probs, top_ids)
        ]

    def classify(self, node_feat):
        subjects = self.predict_topk(
            node_feat,
            self.assets.subject_model,
            self.assets.subject_labels
        )
        grades = self.predict_topk(
            node_feat,
            self.assets.grade_model,
            self.assets.grade_labels
        )
        return subjects, grades
