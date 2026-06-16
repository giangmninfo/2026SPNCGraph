from abc import ABC, abstractmethod


class IHealthService(ABC):

    @abstractmethod
    def check(self) -> None:
        """Perform a health check."""
        pass