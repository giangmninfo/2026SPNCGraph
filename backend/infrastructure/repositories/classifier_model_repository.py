# backend/infrastructure/repositories/classifier_model_repository.py

import json
import torch
from pathlib import Path

from backend.interfaces.repositories.classifier_model_repository_interface import IClassifierModelRepository
from backend.infrastructure.ml.models.graphsage import GraphSAGE
from backend.infrastructure.ml.models.assets import GraphSAGEAssets


class FileSystemClassifierModelRepository(IClassifierModelRepository):
    def __init__(self, artifact_dir: Path):
        """
        artifact_dir example:
        backend/infrastructure/ml/artifacts/v2
        """
        self.artifacts = artifact_dir

    def load_assets(self) -> GraphSAGEAssets:
        graph = torch.load(
            self.artifacts / "graph_data.pt",
            map_location="cpu"
        )

        subject_labels = self._load_json("subject_labels.json")
        grade_labels = self._load_json("grade_labels.json")

        model_subject = self._load_model(
            self.artifacts / "graphsage_subject.pt",
            len(subject_labels)
        )

        model_grade = self._load_model(
            self.artifacts / "graphsage_grade.pt",
            len(grade_labels)
        )

        node_embeddings = self._load_node_embeddings()

        return GraphSAGEAssets(
            x=graph["x"],
            edge_index=graph["edge_index"],
            subject_model=model_subject,
            grade_model=model_grade,
            subject_labels=subject_labels,
            grade_labels=grade_labels,
            node_embeddings=node_embeddings
        )

    def _load_json(self, name: str):
        with open(self.artifacts / name, encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self, path: Path, out_dim: int):
        model = GraphSAGE(896, 256, out_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    
    def _load_node_embeddings(self):
        path = self.artifacts / "node_embeddings.pt"
        if not path.exists():
            return None  # backward compatible

        return torch.load(path, map_location="cpu")
