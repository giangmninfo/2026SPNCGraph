import torch
from PIL import Image
import open_clip
from pathlib import Path
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

from backend.settings import OPEN_CLIP_MODEL_DIR


class CLIPImageEncoder:
    """
    Image encoder using OpenCLIP ViT-B-32 (512-dim).
    """

    def __init__(self, device="cpu"):
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="openai"  # downloadable
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        feat = self.model.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        return feat.cpu()  # (1, 512)

class ResNetImageEncoder:
    """
    Image encoder using ResNet50 backbone.

    Output:
        Tensor shape (1, 2048)
    """

    def __init__(self, device="cpu"):
        self.device = device

        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()  # remove classifier head
        model.to(self.device)
        model.eval()

        self.model = model

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def encode(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        feat = self.model(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        return feat.cpu()  # (1, 2048)