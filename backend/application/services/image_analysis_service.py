# services/image_analysis_service.py
from io import BytesIO
from PIL import Image

class ImageAnalysisService:
    @staticmethod
    def analyze(image_bytes: bytes) -> dict:
        with Image.open(BytesIO(image_bytes)) as img:
            width, height = img.size
            fmt = img.format  # 'PNG', 'JPEG', 'JFIF', etc.

        return {
            "image_size": f"{width}×{height}",
            "image_width": width,
            "image_height": height,
            "image_format": fmt,
        }
