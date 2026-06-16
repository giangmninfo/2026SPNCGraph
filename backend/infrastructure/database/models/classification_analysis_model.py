# backend/infrastructure/database/models/classification_analysis.py

from sqlalchemy import Column, Integer, Float, Text, TIMESTAMP
from sqlalchemy.sql import func
from backend.infrastructure.database.base import Base


class ClassificationAnalysisModel(Base):
    __tablename__ = "classification_analyses"

    id = Column(Integer, primary_key=True, index=True)

    public_code = Column(Text, nullable=False, unique=True, index=True)

    user_id = Column(Integer, nullable=False)

    image_path = Column(Text, nullable=False)

    label = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)

    subject = Column(Text, nullable=False)
    subject_code = Column(Text, nullable=False)
    grade = Column(Integer, nullable=False)

    model_variant = Column(Text, nullable=False)

    created_at = Column(TIMESTAMP, server_default=func.now())
