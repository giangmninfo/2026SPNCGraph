from pathlib import Path
import easyocr
import torch


class OCRReader:
    def __init__(self):
        self.reader = easyocr.Reader(
            ["vi", "en"],
            gpu=torch.cuda.is_available()
        )

    def extract(self, image_path, min_conf=0.4) -> str:
        image_path = Path(image_path)
        results = self.reader.readtext(str(image_path))

        texts = [
            text for _, text, conf in results
            if conf >= min_conf
        ]
        
        extracted_text = " ".join(texts).strip()
        print(extracted_text)

        return extracted_text
