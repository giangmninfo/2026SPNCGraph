# backend/interfaces/repositories/classifier_model_repository_interface.py

from abc import ABC, abstractmethod
from backend.infrastructure.ml.models.assets import GraphSAGEAssets


class IClassifierModelRepository(ABC):
    @abstractmethod
    def load_assets(self) -> GraphSAGEAssets:
        """
        Load and return all assets required for classification:
        - graph tensors
        - models
        - labels
        """
        pass
