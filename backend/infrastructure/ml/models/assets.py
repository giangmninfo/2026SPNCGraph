# infrastructure/ml/models/graphsage_assets.py

class GraphSAGEAssets:
    def __init__(
        self,
        x,
        edge_index,
        subject_model,
        grade_model,
        subject_labels,
        grade_labels,
        node_embeddings = None
    ):
        self.x = x
        self.edge_index = edge_index
        self.subject_model = subject_model
        self.grade_model = grade_model
        self.subject_labels = subject_labels
        self.grade_labels = grade_labels
        self.node_embeddings = node_embeddings


    def load_assets(self):
        x, edge_index = self.get_graph()
        labels = self.get_labels()

        return GraphSAGEAssets(
            x=x,
            edge_index=edge_index,
            subject_model=self.get_subject_classifier(),
            grade_model=self.get_grade_classifier(),
            subject_labels=labels["subject"],
            grade_labels=labels["grade"],
            node_embeddings=self.get_node_embeddings()  # 👈
        )