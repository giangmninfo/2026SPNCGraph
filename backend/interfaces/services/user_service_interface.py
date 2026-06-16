from abc import ABC, abstractmethod
from backend.domain.entities.user import User


class IUserService(ABC):

    # ====================
    #  AUTHENTICATION
    # ====================
    @abstractmethod
    def authenticate(self, username: str, password: str) -> User:
        """Authenticate a user by username"""
        pass

    # ====================
    #  USER MANAGEMENT
    # ====================
    @abstractmethod
    def create_user(self, user: User) -> User:
        """Create a new user."""
        pass

    @abstractmethod
    def get_user_by_identifier(self, username: str) -> User | None:
        """Retrieve a user by username or email."""
        pass

    @abstractmethod
    def is_username_available(self, username: str) -> bool:
        """Check if a username is available."""
        pass

    @abstractmethod
    def get_user_by_username(self, username: str) -> User | None:
        pass

    @abstractmethod
    def get_user_by_id(self, user_id: int) -> User | None:
        pass

    @abstractmethod
    def list_users(self) -> list[User]:
        pass