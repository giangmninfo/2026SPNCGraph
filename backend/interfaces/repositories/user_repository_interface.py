from abc import ABC, abstractmethod
from typing import Optional, List
from backend.domain.entities.user import User

class IUserRepository(ABC):
    @abstractmethod
    def create(self, user: User) -> User:
        """
        Persist a user and return the saved user
        (e.g., with ID assigned)
        """
        pass

    @abstractmethod
    def get_by_email(self, email: str) -> Optional[User]:
        pass

    @abstractmethod
    def get_by_username(self, username: str) -> Optional[User]:
        """
        Return a user if username exists, otherwise None
        """
        pass

    @abstractmethod
    def get_by_id(self, user_id: int) -> Optional[User]:
        """
        Find a user by ID
        """
        pass

    @abstractmethod
    def list_all(self) -> List[User]:
        """
        Return all users
        """
        pass
