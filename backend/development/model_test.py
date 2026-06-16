from backend.infrastructure.database.postgres import SessionLocal
from backend.infrastructure.database.models.user_model import UserModel

session = SessionLocal()

users = session.query(UserModel).all()
print(users)
