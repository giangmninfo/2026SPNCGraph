import re
from werkzeug.security import generate_password_hash, check_password_hash

from backend.interfaces.services.user_service_interface import IUserService
from backend.interfaces.repositories.user_repository_interface import IUserRepository
from backend.domain.entities.user import User
from backend.domain.exception import InvalidCredentials, InvalidUserData

USERNAME_REGEX = re.compile(r"^[a-z0-9]{3,20}$")

class UserService(IUserService):
    def __init__(self, user_repository: IUserRepository):
        self.user_repository = user_repository

    #====================
    #  PRIVATE CLASSES
    #====================
    def _validate_username(self, username: str):
        if not username:
            raise InvalidUserData("Username is required")

        if not USERNAME_REGEX.match(username):
            raise InvalidUserData(
                "Username must be lowercase, alphanumeric, no spaces"
            )

    def _normalize_user(self, user: User):
        user.username = user.username.strip().lower()
        user.email = user.email.strip().lower()
        user.full_name = user.full_name.strip()
    #====================
    #  PUBLIC CLASSES
    #====================
    def authenticate(self, username: str, password: str) -> User:
        user = self.user_repository.get_by_username(username)

        if not user:
            raise InvalidCredentials("Invalid credentials")

        if not check_password_hash(user.password.decode(), password):
            raise InvalidCredentials("Invalid credentials")

        return user

    def create_user(self, user: User) -> User:
        self._normalize_user(user)
        self._validate_username(user.username)

        if not self.is_username_available(user.username):
            raise InvalidUserData("Username already exists")

        user.password = generate_password_hash(user.password).encode()
        return self.user_repository.create(user)
    
    def get_user_by_identifier(self, identifier: str):
        if "@" in identifier:
            return self.user_repository.get_by_email(identifier)
        return self.user_repository.get_by_username(identifier)
    
    def is_username_available(self, username: str) -> bool:
        user = self.user_repository.get_by_username(username)
        return user is None

    def get_user_by_username(self, username: str):
        return self.user_repository.get_by_username(username)

    def get_user_by_id(self, user_id: int):
        return self.user_repository.get_by_id(user_id)

    def list_users(self):
        return self.user_repository.list_all()
 