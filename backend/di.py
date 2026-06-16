from pathlib import Path

from backend.settings import ARTIFACTS_DIR, ARTIFACTS_VERSION

from backend.application.services.health_service import HealthService
from backend.infrastructure.repositories.postgres_health_repository import PostgresHealthRepository

from backend.application.services.user_service import UserService
from backend.infrastructure.repositories.postgres_user_repository import PostgresUserRepository

from backend.application.services.classification_history_service import ClassificationHistoryService
from backend.infrastructure.repositories.classification_analysis_repository import PostgresClassificationAnalysisRepository 

from backend.application.services.image_storage_service import SupabaseImageStorageService
from backend.infrastructure.database.supabase import create_supabase_client

from backend.application.services.image_analysis_service import ImageAnalysisService

from backend.infrastructure.ml.ocr.ocr_engine import OCRReader
from backend.infrastructure.ml.encoders.image_encoder import CLIPImageEncoder
from backend.infrastructure.ml.encoders.text_encoder import MiniLML6TextEncoder
from backend.infrastructure.ml.encoders.graphsage_encoder import GraphSAGEEncoder
from backend.infrastructure.ml.features.clip_image_feature import CLIPImageFeatureBuilder
from backend.infrastructure.ml.features.node_feature_builder import NodeFeatureBuilder
from backend.infrastructure.ml.features.text_feature_builder import TextFeatureBuilder
from backend.infrastructure.ml.classifiers.graphsage_classifier import GraphSAGEClassifier
from backend.infrastructure.ml.classifiers.knn_classifier import KNNClassifier, GraphSAGEKNNClassifier
from backend.infrastructure.ml.similarity.neighbor_search import NeighborSearcher
from backend.infrastructure.ml.image_classifier import VotingImageClassifier, GraphSAGEImageClassifier, GraphSAGEKNNImageClassifier
from backend.infrastructure.repositories.classifier_model_repository import FileSystemClassifierModelRepository

from backend.application.services.image_classifier_service import ImageClassificationService
from backend.settings import ARTIFACTS_DIR, ARTIFACTS_VERSION

from backend.infrastructure.repositories.voting_model_repository import FileSystemVotingModelRepository
from backend.infrastructure.ml.classifiers.voting_classifier import VotingClassifier
from backend.infrastructure.ml.encoders.image_encoder import ResNetImageEncoder
from backend.infrastructure.ml.encoders.text_encoder import MiniLML12TextEncoder

from backend.infrastructure.repositories.knn_graphsage_classifier_model_repository import KNNGraphSAGEModelRepository

health_service = HealthService(PostgresHealthRepository())
user_service = UserService(PostgresUserRepository())
classification_history_service = ClassificationHistoryService(PostgresClassificationAnalysisRepository())
image_storage_service = SupabaseImageStorageService(create_supabase_client(), "GNN Classifier Image Storage")
image_analysis_service = ImageAnalysisService()

# Load assets
# artifact_dir = ARTIFACTS_DIR / ARTIFACTS_VERSION
artifact_dir = ARTIFACTS_DIR / "GNN_single_v1"

ocr = OCRReader()

# GNN_SINGLE_V1
repo = FileSystemVotingModelRepository(artifact_dir)
assets = repo.load_assets()

print("Voting feature DB shape:", assets.features.shape)
print("Metadata rows:", len(assets.metadata))

image_encoder = ResNetImageEncoder(device="cpu")   # (1, 2048)
text_encoder = MiniLML12TextEncoder()              # (1, 384)

image_feature_builder = CLIPImageFeatureBuilder(image_encoder)
text_feature_builder = TextFeatureBuilder(ocr, text_encoder)

node_feature_builder = NodeFeatureBuilder(
    image_feature_builder,
    text_feature_builder
)

voting_classifier = VotingClassifier(
    features_db=assets.features,
    metadata_df=assets.metadata
)

image_classifier = VotingImageClassifier(
    node_feature_builder=node_feature_builder,
    voting_classifier=voting_classifier
)

image_encoder = ResNetImageEncoder(device="cpu")   # (1, 2048)
text_encoder = MiniLML12TextEncoder()              # (1, 384)

image_feature_builder = CLIPImageFeatureBuilder(image_encoder)
text_feature_builder = TextFeatureBuilder(ocr, text_encoder)

node_feature_builder = NodeFeatureBuilder(
    image_feature_builder,
    text_feature_builder
)

image_classifier = VotingImageClassifier(
    node_feature_builder=node_feature_builder,
    voting_classifier=voting_classifier
)

voting_classifier = VotingClassifier(
    features_db=assets.features,
    metadata_df=assets.metadata,
    top_k=10,
)

# GNN_DUAL_V2 (OLD, LEGACY)
artifact_dir_2 = ARTIFACTS_DIR / "GNN_dual_v2"
repo_2 = FileSystemClassifierModelRepository(artifact_dir_2)
assets_2 = repo_2.load_assets()

print("Graph x shape:", assets_2.x.shape)

edge_index = assets_2.edge_index

edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
reverse_edges = set(zip(edge_index[1].tolist(), edge_index[0].tolist()))

print("Missing reverse edges:", len(edges - reverse_edges))


neighbor_searcher = NeighborSearcher(assets_2.x.numpy())
graphsage_classifier = GraphSAGEClassifier(assets_2, neighbor_searcher)

clip_encoder = CLIPImageEncoder(device="cpu")
text_dual_encoder = MiniLML6TextEncoder()

image_feature_dual_builder = CLIPImageFeatureBuilder(clip_encoder)
text_feature_dual_builder = TextFeatureBuilder(ocr, text_dual_encoder)

node_dual_feature_builder = NodeFeatureBuilder(
    image_feature_dual_builder,
    text_feature_dual_builder
)

image_dual_classifier = GraphSAGEImageClassifier(
    node_feature_builder=node_dual_feature_builder,
    graphsage_classifier=graphsage_classifier
)

# knn

encoder = GraphSAGEEncoder()
encoder.conv1.load_state_dict(
    assets_2.subject_model.conv1.state_dict()
)

node_emb = assets_2.node_embeddings

subject_knn = KNNClassifier(
    node_emb["embeddings"].numpy(),
    node_emb["subject_labels"]
)

grade_knn = KNNClassifier(
    node_emb["embeddings"].numpy(),
    node_emb["grade_labels"]
)

graphsage_knn_classifier = GraphSAGEKNNClassifier(
    encoder=encoder,
    subject_knn=subject_knn,
    grade_knn=grade_knn,
    subject_labels=assets_2.subject_labels,
    grade_labels=assets_2.grade_labels
)

image_knn_classifier = GraphSAGEKNNImageClassifier(
    node_feature_builder=node_dual_feature_builder,
    knn_classifier=graphsage_knn_classifier
)


# knn-GNN
# repo_3 = KNNGraphSAGEModelRepository(artifact_dir)
# assets_3 = repo_3.load_assets()

# print("Graph x shape:", assets_3.x.shape)

# neighbor_searcher_3 = NeighborSearcher(assets_3.x.numpy())
# graphsage_classifier_3 = KNNGraphSAGEClassifier(assets_3, neighbor_searcher_3)

# knn_graphsage_classifier = KNNGraphSAGEClassifier(
#     assets=assets_3,
#     neighbor_searcher=neighbor_searcher_3,
#     k_neighbors=10,
#     topk=3,
# )

# graphsage_classifier_single = KNNGraphSAGEImageClassifier(
#     node_feature_builder=node_feature_builder,
#     graphsage_classifier=knn_graphsage_classifier,
# )


# => Sub / Gra
# # => Train Sub, Train Grade => 2 đầu => .pt
# => .pt .pt
# => Sub x Grade => 

# => Sub x Gra (X)
# => .pt (.xlsx)
# => Voting (cosine sim) => 

# Infra orchestrator
image_classifier_service = ImageClassificationService(
    single_classifier=image_classifier,
    dual_classifier=image_dual_classifier,
    single_graphsage_classifier=image_knn_classifier
)