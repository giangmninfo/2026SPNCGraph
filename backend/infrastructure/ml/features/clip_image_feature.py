# backend/infrastructure/ml/features/clip_image_feature.py

import torch
from pathlib import Path

class CLIPImageFeatureBuilder:
    def __init__(self, image_encoder):
        self.image_encoder = image_encoder

    def build(self, image_path: Path) -> torch.Tensor:
        image_path = Path(image_path)
        return self.image_encoder.encode(image_path)  # (1, 512)
