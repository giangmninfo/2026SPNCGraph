import torch
import pandas as pd
from pathlib import Path
from torch.serialization import add_safe_globals

from backend.infrastructure.ml.models.knn_graphsage_assets import KNNGraphSAGEAssets

# 🔐 Allow torch_geometric objects (trusted artifact)
from torch_geometric.data.data import Data, DataEdgeAttr


class KNNGraphSAGEModelRepository:
    """
    Loads assets for inductive kNN GraphSAGE classifier.

    Expected structure:
        GNN_single_v1/
            graph_data_multimodal.pt
            metadata.csv
    """

    def __init__(self, artifact_dir: Path):
        self.artifacts = artifact_dir

    def load_assets(self) -> KNNGraphSAGEAssets:
        # ✅ Allowlist PyG globals (PyTorch ≥ 2.6)
        add_safe_globals([Data, DataEdgeAttr])

        graph = torch.load(
            self.artifacts / "graph_data_multimodal.pt",
            map_location="cpu",
            weights_only=False,   # 🔴 REQUIRED for full object graph
        )
        print(graph)
        print(graph.keys)

        x = graph["x"]                     # torch.Tensor (N, D)
        model = graph["model"]             # GraphSAGE nn.Module
        label_map = graph["label_map"]     # {int: "SUBJECT - GRADE"}

        metadata = pd.read_csv(
            self.artifacts / "metadata.csv",
            encoding="utf-8"
        )

        assert x.size(0) == len(metadata), \
            "Feature DB and metadata row count mismatch"

        return KNNGraphSAGEAssets(
            x=x,
            metadata=metadata,
            model=model,
            inv_label_map=label_map,
        )
