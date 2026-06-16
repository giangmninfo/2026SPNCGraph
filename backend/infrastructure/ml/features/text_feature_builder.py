# backend/infrastructure/ml/features/text_feature_builder.py

from pathlib import Path
import torch

class TextFeatureBuilder:
    def __init__(self, ocr_reader, text_encoder):
        self.ocr = ocr_reader
        self.encoder = text_encoder

    def build(self, image_path: Path) -> torch.Tensor:
        text = self.ocr.extract(Path(image_path))
        return self.encoder.encode(text)
