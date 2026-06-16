from datetime import datetime
from typing import Optional


class ClassificationAnalysis:
    def __init__(
        self,
        *,
        id: Optional[int] = None,
        public_code: Optional[str] = None,
        user_id: int,
        image_path: str,
        label: str,
        confidence: float,
        subject: str,
        subject_code: str,
        grade: int,
        model_variant: str,
        created_at: Optional[datetime] = None,
    ):
        if not user_id:
            raise ValueError("user_id is required")

        if not image_path:
            raise ValueError("image_path is required")

        if not label:
            raise ValueError("label is required")

        if not (0 <= confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")

        self.id = id
        self.public_code = public_code
        self.user_id = user_id
        self.image_path = image_path
        self.label = label
        self.confidence = confidence
        self.subject = subject
        self.subject_code = subject_code
        self.grade = grade
        self.model_variant = model_variant
        self.created_at = created_at