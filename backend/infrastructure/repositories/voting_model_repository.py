# backend/infrastructure/repositories/voting_model_repository.py

import torch
import pandas as pd
from pathlib import Path

from backend.infrastructure.ml.models.voting_assets import VotingAssets


class FileSystemVotingModelRepository:
    """
    Loads assets for similarity-based voting classifier
    from a frozen multimodal graph artifact.

    Expected structure:
        GNN_single_v1/
            graph_data_multimodal.pt
            metadata.csv
    """

    def __init__(self, artifact_dir: Path):
        self.artifacts = artifact_dir

    def load_assets(self) -> VotingAssets:
        graph = torch.load(
            self.artifacts / "graph_data_multimodal.pt",
            map_location="cpu",
            weights_only=False,   # ← REQUIRED
        )

        features = graph["x"].cpu().numpy()  # (N, 2432)

        metadata = pd.read_csv(
            self.artifacts / "metadata.csv",
            encoding="utf-8"
        )

        assert len(features) == len(metadata), \
            "Feature DB and metadata row count mismatch"

        return VotingAssets(
            features=features,
            metadata=metadata
        )
