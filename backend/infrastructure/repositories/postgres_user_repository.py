from typing import Optional, List
from sqlalchemy.exc import IntegrityError

from backend.domain.entities.user import User
from backend.interfaces.repositories.user_repository_interface import IUserRepository

from backend.infrastructure.database.postgres import SessionLocal
from backend.infrastructure.database.models.user_model import UserModel

class PostgresUserRepository(IUserRepository):

    def _to_entity(self, model: UserModel) -> User:
        return User(
            id=model.id,
            full_name=model.full_name,
            username=model.username,
            email=model.email,
            password=model.password,
            avatar_color=model.avatar_color,
            created_at=model.created_at,
        )

    def _to_model(self, user: User) -> UserModel:
        return UserModel(
            full_name=user.full_name,
            username=user.username,
            email=user.email,
            password=user.password,
            avatar_color=user.avatar_color,
        )

    def create(self, user: User) -> User:
        db = SessionLocal()
        try:
            model = self._to_model(user)
            db.add(model)
            db.commit()
            db.refresh(model)
            return self._to_entity(model)
        finally:
            db.close()

    def get_by_id(self, user_id: int):
        db = SessionLocal()
        try:
            model = db.query(UserModel).filter_by(id=user_id).first()
            return self._to_entity(model) if model else None
        finally:
            db.close()

    def get_by_username(self, username: str):
        db = SessionLocal()
        try:
            model = db.query(UserModel).filter_by(username=username).first()
            return self._to_entity(model) if model else None
        finally:
            db.close()

    def get_by_email(self, email: str):
        db = SessionLocal()
        try:
            model = db.query(UserModel).filter_by(email=email).first()
            return self._to_entity(model) if model else None
        finally:
            db.close()

    def list_all(self) -> List[User]:
        db = SessionLocal()
        try:
            models = db.query(UserModel).all()
            return [self._to_entity(m) for m in models]
        finally:
            db.close()
