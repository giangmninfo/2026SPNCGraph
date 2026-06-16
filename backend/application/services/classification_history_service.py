# backend/application/services/classification_history_service.py
import math
from typing import List

from backend.domain.entities.classification_analysis import ClassificationAnalysis
from backend.interfaces.repositories.classification_analysis_repository_interface import IClassificationAnalysisRepository
from backend.interfaces.services.classification_history_service_interface import IClassificationHistoryService

from backend.infrastructure.database.postgres import SessionLocal


class ClassificationHistoryService(IClassificationHistoryService):

    DEFAULT_PAGE_SIZE = 3

    def __init__(self, repository: IClassificationAnalysisRepository):
        self.repository = repository

    def save(
        self,
        *,
        user_id: int,
        image_path: str,
        result: dict,
        model_variant: str,
    ) -> ClassificationAnalysis:

        db = SessionLocal()
        try:
            # 1️⃣ Normalize subject prefix (ENG, MAT, DEF, ...)
            subject_enum = result["subject_code"]  # e.g. "ENGLISH"
            subject_prefix = subject_enum.replace("_", "")[:3]

            # 2️⃣ Get next number per subject prefix
            next_number = self.repository.get_next_subject_number(
                subject_code=subject_prefix,
                db=db,
            )

            # 3️⃣ Build public code
            public_code = f"{subject_prefix}-{next_number:03d}"

            analysis = ClassificationAnalysis(
                id=None,
                public_code=public_code,
                user_id=user_id,
                image_path=image_path,
                label=result["label"],
                confidence=float(result["confidence"]),
                subject=result["subject"],
                subject_code=subject_enum,  # store full enum
                grade=int(result["grade"]),
                model_variant=model_variant,
            )

            saved = self.repository.create(analysis, db=db)
            db.commit()
            return saved

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # def save(self, *, user_id: int, image_path: str, result: dict, model_variant: str) -> ClassificationAnalysis:
    #     """
    #     Persist only the PRIMARY prediction from model output.
    #     """

    #     analysis = ClassificationAnalysis(
    #         id=None,
    #         user_id=user_id,
    #         image_path=image_path,
    #         label=result["label"],
    #         confidence=float(result["confidence"]),
    #         subject=result["subject"],
    #         subject_code=result["subject_code"],
    #         grade=int(result["grade"]),
    #         model_variant=model_variant,
    #     )

    #     return self.repository.create(analysis)

    def list_user_history(
        self,
        *,
        user_id: int,
        page: int = 1,
        q: str | None = None,
    ):
        items, total = self.repository.list_by_user(
            user_id=user_id,
            page=page,
            limit=self.DEFAULT_PAGE_SIZE,
            q=q,
        )

        return {
            "items": items,
            "page": page,
            "limit": self.DEFAULT_PAGE_SIZE,
            "total": total,
            "total_pages": math.ceil(total / self.DEFAULT_PAGE_SIZE),
        }