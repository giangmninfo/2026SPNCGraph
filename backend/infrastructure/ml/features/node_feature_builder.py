# backend/infrastructure/ml/features/node_feature_builder.py

import torch
from pathlib import Path

class NodeFeatureBuilder:
    def __init__(self, image_feature_builder, text_feature_builder):
        self.image_builder = image_feature_builder
        self.text_builder = text_feature_builder

    def build(self, image_path: Path) -> torch.Tensor:
        img_feat = self.image_builder.build(image_path)   # (1, 512)
        text_feat = self.text_builder.build(image_path)   # (1, 384)

        return torch.cat([text_feat, img_feat], dim=1)    # (1, 896)
