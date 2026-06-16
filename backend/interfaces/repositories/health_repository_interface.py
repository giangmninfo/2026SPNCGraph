from abc import ABC, abstractmethod

class IHealthRepository(ABC):

    @abstractmethod
    def check(self) -> None:
        pass
