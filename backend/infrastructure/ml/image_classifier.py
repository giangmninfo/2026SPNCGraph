# backend/infrastructure/ml/image_classifier.py

import tempfile
from pathlib import Path
from backend.infrastructure.ml.postprocess.subject_grade_combiner import combine_subject_grade


class GraphSAGEImageClassifier:
    """
    ⚠️ DEPRECATED (LEGACY – GNN DUAL PIPELINE)

    This classifier implements the legacy GraphSAGE-based *dual-head*
    (subject + grade) inference pipeline.

    Status:
        - Frozen for backward compatibility
        - Used by: ImageClassificationService.classify_image_dual(...)
        - NOT recommended for new development

    Reason for deprecation:
        - Strong coupling to graph artifacts and node feature shapes (896-dim)
        - Incompatible with newer single-head / voting-based pipelines
        - Difficult to retrain or extend safely under deadline constraints

    Migration path:
        - Use VotingImageClassifier (single-head / similarity-based)
        - Service layer remains stable; only infra classifier changes

    ⚠️ Do NOT extend or modify this class unless maintaining legacy artifacts.
    """

    def __init__(
        self,
        node_feature_builder,
        graphsage_classifier,
    ):
        self.node_feature_builder = node_feature_builder
        self.classifier = graphsage_classifier

    def classify(self, image_bytes: bytes) -> dict:
        """
        Legacy inference entrypoint.

        Returns:
            {
                "subjects": Dict[str, float],
                "grades": Dict[str, float]
            }

        NOTE:
            This method exists only to support legacy deployments.
            New classifiers must implement IImageClassifierInterface
            and return raw scores in a model-agnostic way.
        """
        image_path = self._save_temp_image(image_bytes)

        node_feat = self.node_feature_builder.build(image_path)

        subjects, grades = self.classifier.classify(node_feat)

        return {
            "subjects": subjects,
            "grades": grades,
        }

    def _save_temp_image(self, image_bytes: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.close()
        return Path(tmp.name)

class VotingImageClassifier:
    """
    Image-level orchestrator for similarity-based voting classifier.
    """

    def __init__(self, node_feature_builder, voting_classifier):
        self.node_feature_builder = node_feature_builder
        self.classifier = voting_classifier

    def classify(self, image_bytes: bytes) -> dict:
        image_path = self._save_temp_image(image_bytes)

        node_feat = self.node_feature_builder.build(image_path)

        # 🔑 returns ONE dict
        return self.classifier.classify(node_feat)

    def _save_temp_image(self, image_bytes: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.close()
        return Path(tmp.name)
    
class GraphSAGEKNNImageClassifier:
    """
    kNN-based replacement for legacy GraphSAGEImageClassifier.

    This classifier:
        - Uses GraphSAGEEncoder to project node features
        - Performs kNN lookup over frozen node embeddings
        - Outputs calibrated confidence scores

    Contract:
        - Fully compatible with ImageClassificationService
        - Same return schema as legacy dual-head classifier
    """

    def __init__(
        self,
        node_feature_builder,
        knn_classifier,
    ):
        self.node_feature_builder = node_feature_builder
        self.classifier = knn_classifier

    def classify(self, image_bytes: bytes) -> dict:
        image_path = self._save_temp_image(image_bytes)

        node_feat = self.node_feature_builder.build(image_path)

        subjects, grades = self.classifier.classify(node_feat)

        return {
            "subjects": subjects,
            "grades": grades,
        }

    def _save_temp_image(self, image_bytes: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.close()
        return Path(tmp.name)


# class KNNGraphSAGEImageClassifier:
#     """
#     Image-level orchestrator for inductive GraphSAGE classifier
#     using kNN-based ego-graph construction.

#     This is the NEW single-head GNN pipeline.
#     """

#     def __init__(self, node_feature_builder, graphsage_classifier):
#         self.node_feature_builder = node_feature_builder
#         self.classifier = graphsage_classifier

#     def classify(self, image_bytes: bytes) -> dict:
#         image_path = self._save_temp_image(image_bytes)

#         node_feat = self.node_feature_builder.build(image_path)

#         # 🔑 returns List[(label, prob)]
#         predictions = self.classifier.classify(node_feat)

#         return {
#             "predictions": [
#                 {"label": label, "score": score}    
#                 for label, score in predictions
#             ]
#         }

#     def _save_temp_image(self, image_bytes: bytes) -> Path:
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#         tmp.write(image_bytes)
#         tmp.close()
#         return Path(tmp.name)
