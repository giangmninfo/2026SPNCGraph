from sqlalchemy import Column, Integer, Text
from backend.infrastructure.database.base import Base

class SubjectSequenceModel(Base):
    __tablename__ = "subject_sequences"

    subject_code = Column(Text, primary_key=True)
    last_number = Column(Integer, nullable=False)
