# backend/interfaces/repositories/classification_analysis_repository_interface.py

from typing import List
from backend.domain.entities.classification_analysis import ClassificationAnalysis


class IClassificationAnalysisRepository:
    def create(self, analysis: ClassificationAnalysis, db) -> ClassificationAnalysis:
        raise NotImplementedError

    def list_by_user(self, user_id: int, page: int, limit: int, q: str) -> List[ClassificationAnalysis]:
        raise NotImplementedError

    def get_next_subject_number(
        self,
        *,
        subject_code: str,
        db,
    ) -> int:
        raise NotImplementedError