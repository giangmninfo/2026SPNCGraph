# backend/interfaces/services/classification_history_service_interface.py

from typing import List
from backend.domain.entities.classification_analysis import ClassificationAnalysis


class IClassificationHistoryService:
    def save(
        self,
        *,
        user_id: int,
        image_path: str,
        result: dict,
        model_variant: str,
    ) -> ClassificationAnalysis:
        raise NotImplementedError

    def list_by_user(
        self,
        user_id: int,
    ) -> List[ClassificationAnalysis]:
        raise NotImplementedError
