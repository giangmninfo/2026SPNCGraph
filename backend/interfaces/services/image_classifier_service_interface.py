from abc import ABC, abstractmethod


class IImageClassifierInterface(ABC):
    @abstractmethod
    def classify(self, image_bytes: bytes) -> dict:
        pass
