# backend/infrastructure/repositories/classification_analysis_repository.py

from typing import List

from sqlalchemy.exc import IntegrityError

from backend.domain.entities.classification_analysis import ClassificationAnalysis
from backend.interfaces.repositories.classification_analysis_repository_interface import IClassificationAnalysisRepository

from backend.infrastructure.database.postgres import SessionLocal
from backend.infrastructure.database.models.classification_analysis_model import ClassificationAnalysisModel
from backend.infrastructure.database.models.subject_sequence_model import SubjectSequenceModel


class PostgresClassificationAnalysisRepository(IClassificationAnalysisRepository):
    def _to_entity(
        self,
        model: ClassificationAnalysisModel,
    ) -> ClassificationAnalysis:
        return ClassificationAnalysis(
            id=model.id,
            public_code=model.public_code,
            user_id=model.user_id,
            image_path=model.image_path,
            label=model.label,
            confidence=model.confidence,
            subject=model.subject,
            subject_code=model.subject_code,
            grade=model.grade,
            model_variant=model.model_variant,
            created_at=model.created_at,
        )

    def _to_model(
        self,
        analysis: ClassificationAnalysis,
    ) -> ClassificationAnalysisModel:
        return ClassificationAnalysisModel(
            public_code=analysis.public_code,
            user_id=analysis.user_id,
            image_path=analysis.image_path,
            label=analysis.label,
            confidence=analysis.confidence,
            subject=analysis.subject,
            subject_code=analysis.subject_code,
            grade=analysis.grade,
            model_variant=analysis.model_variant,
        )

    def create(
        self,
        analysis: ClassificationAnalysis,
        db
    ) -> ClassificationAnalysis:
        try:
            model = self._to_model(analysis)
            db.add(model)
            db.commit()
            db.flush()
            db.refresh(model)
            return self._to_entity(model)
        finally:
            db.close()

    def list_by_user(
        self,
        user_id: int,
        *,
        page: int = 1,
        limit: int = 3,
        q: str | None = None,
    ):
        offset = (page - 1) * limit
        db = SessionLocal()
        try:
            query = (
                db.query(ClassificationAnalysisModel)
                .filter(ClassificationAnalysisModel.user_id == user_id)
            )

            if q:
                query = query.filter(
                    ClassificationAnalysisModel.public_code.ilike(f"%{q}%")
                    | ClassificationAnalysisModel.subject.ilike(f"%{q}%")
                )

            total = query.count()

            models = (
                query
                .order_by(ClassificationAnalysisModel.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            return [self._to_entity(m) for m in models], total

        finally:
            db.close()


    def get_next_subject_number(
        self,
        *,
        subject_code: str,
        db,
    ) -> int:
        seq = (
            db.query(SubjectSequenceModel)
            .filter_by(subject_code=subject_code)
            .with_for_update()
            .first()
        )

        if not seq:
            seq = SubjectSequenceModel(
                subject_code=subject_code,
                last_number=1,
            )
            db.add(seq)
            return 1

        seq.last_number += 1
        return seq.last_number