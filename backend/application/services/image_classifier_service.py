# backend/application/services/image_classifier_service.py
import time
import traceback
import math

from backend.application.exception import MLServiceUnavailable
from backend.interfaces.services.image_classifier_service_interface import IImageClassifierInterface
from backend.domain.value_objects.subject import SubjectCode, SUBJECT_VI_TO_CODE
from backend.infrastructure.ml.postprocess.subject_grade_combiner import combine_subject_grade

class ImageClassificationService:
    def __init__(
        self,
        single_classifier: IImageClassifierInterface,
        dual_classifier: IImageClassifierInterface,
        single_graphsage_classifier: IImageClassifierInterface
    ):
        self.single_classifier = single_classifier
        self.dual_classifier = dual_classifier
        self.single_graphsage_classifier = single_graphsage_classifier

    @staticmethod
    def _split_subject_grade(label: str) -> tuple[str, int]:
        subject, grade = label.rsplit(" - ", 1)
        return subject, int(grade)
    
    @staticmethod
    def _subject_code(subject: str) -> SubjectCode:
        return SUBJECT_VI_TO_CODE.get(subject, SubjectCode.UNKNOWN)

    @staticmethod
    def _softmax(items: dict) -> dict:
        values = list(items.values())
        max_v = max(values)

        exps = {k: math.exp(v - max_v) for k, v in items.items()}
        total = sum(exps.values())

        return {k: v / total for k, v in exps.items()}

    def classify_image_dual(self, image_bytes: bytes) -> dict:
        start_time = time.perf_counter()
        
        try:
            raw = self.dual_classifier.classify(image_bytes)
        except Exception as e:
            print("🔥 ML ROOT ERROR:")
            traceback.print_exc()
            raise MLServiceUnavailable("ML inference failed") from e

        subjects = raw["subjects"]
        grades = raw["grades"]

        # 1. Combine
        pairs = combine_subject_grade(
            subjects,
            grades,
            method="weighted"
        )

        # 2. Sort
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        # 3. Extract primary
        primary_label, primary_conf = pairs[0]

        subject, grade = self._split_subject_grade(primary_label)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # 4. Shape payload
        return {
            "label": primary_label,
            "confidence": primary_conf,
            "subject": subject,
            "subject_code": self._subject_code(subject),
            "grade": grade,
            "processing_time_ms": round(processing_time_ms, 2),
            "model_variant": "GraphSAGE-I_v2",
            "graph_nodes": 9644,
            "graph_edges": 39504,
            "dimension": 896,
            "top_predictions": [
                {
                    "label": label,
                    "confidence": score,
                    "subject": self._split_subject_grade(label)[0],
                    "subject_code": self._subject_code(self._split_subject_grade(label)[0]),
                    "grade": self._split_subject_grade(label)[1],
                }
                for label, score in pairs
            ],
        }

    def classify_image_single(self, image_bytes: bytes) -> dict:
        start_time = time.perf_counter()

        try:
            raw = self.single_classifier.classify(image_bytes)
        except Exception as e:
            print("🔥 ML ROOT ERROR:")
            traceback.print_exc()
            raise MLServiceUnavailable("ML inference failed") from e

        pairs = raw.get("pairs", {})
        if not pairs:
            raise MLServiceUnavailable("No predictions returned")

        # Softmax normalization
        pairs_prob = self._softmax(pairs)

        # Primary prediction
        primary_label, primary_conf = max(
            pairs_prob.items(),
            key=lambda x: x[1]
        )

        subject, grade = self._split_subject_grade(primary_label)

        # Top-k by raw score, displayed by probability
        top_labels = sorted(
            pairs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "label": primary_label,
            "confidence": float(primary_conf),
            "subject": subject,
            "subject_code": self._subject_code(subject),
            "grade": grade,
            "processing_time_ms": round(processing_time_ms, 2),
            "model_variant": "kNN-Voting",
            "graph_nodes": 9644,
            "graph_edges": None,
            "dimension": 2432,
            "top_predictions": [
                {
                    "label": label,
                    "confidence": float(pairs_prob[label]),
                    "subject": self._split_subject_grade(label)[0],
                    "subject_code": self._subject_code(
                        self._split_subject_grade(label)[0]
                    ),
                    "grade": self._split_subject_grade(label)[1],
                }
                for label, _ in top_labels
            ],
        }

    def classify_image_knn_graphsage(self, image_bytes: bytes) -> dict:
        start_time = time.perf_counter()

        try:
            raw = self.single_graphsage_classifier.classify(image_bytes)
        except Exception as e:
            print("🔥 ML ROOT ERROR:")
            traceback.print_exc()
            raise MLServiceUnavailable("ML inference failed") from e

        subjects = raw["subjects"]   # Dict[str, float]
        grades = raw["grades"]       # Dict[str, float]

        print(raw)
        # 🔧 INLINE COMBINER
        pairs = []
        alpha = 0.7
        beta = 0.3

        for subject, ps in subjects.items():
            for grade, pg in grades.items():
                score = (ps ** alpha) * (pg ** beta)
                pairs.append((f"{subject} - {grade}", score))

        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        primary_label, primary_conf = pairs[0]
        subject, grade = self._split_subject_grade(primary_label)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "label": primary_label,
            "confidence": float(primary_conf),
            "subject": subject,
            "subject_code": self._subject_code(subject),
            "grade": grade,
            "processing_time_ms": round(processing_time_ms, 2),
            "model_variant": "GraphSAGE-E_kNN",
            "graph_nodes": 9644,
            "graph_edges": None,
            "dimension": 896,
            "top_predictions": [
                {
                    "label": label,
                    "confidence": float(score),
                    "subject": self._split_subject_grade(label)[0],
                    "subject_code": self._subject_code(
                        self._split_subject_grade(label)[0]
                    ),
                    "grade": self._split_subject_grade(label)[1],
                }
                for label, score in pairs
            ],
        }