from sqlalchemy import (
    Column,
    Integer,
    String,
    LargeBinary,
    TIMESTAMP,
    func
)

from backend.infrastructure.database.base import Base


class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    full_name = Column(String(100), nullable=False)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    password = Column(LargeBinary, nullable=False)
    avatar_color = Column(String(7), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
