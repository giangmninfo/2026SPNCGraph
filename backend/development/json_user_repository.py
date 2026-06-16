import json
from pathlib import Path
from typing import List, Optional

from backend.domain.entities.user import User
from backend.interfaces.repositories.user_repository import UserRepository


class JsonUserRepository(UserRepository):
    def __init__(self, file_path: str = "data/users.json"):
        self.file_path = Path(file_path)
        self._ensure_file_exists()

    # ---------- Private helpers ----------

    def _ensure_file_exists(self) -> None:
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(json.dumps({"users": []}, indent=2))

    def _read_data(self) -> dict:
        with self.file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write_data(self, data: dict) -> None:
        with self.file_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def _to_entity(self, raw: dict) -> User:
        return User(
            id=raw["id"],
            name=raw["name"],
            email=raw["email"]
        )

    def _to_dict(self, user: User) -> dict:
        return {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }

    def _next_id(self, users: List[dict]) -> int:
        if not users:
            return 1
        return max(user["id"] for user in users) + 1

    # ---------- Repository methods ----------

    def save(self, user: User) -> User:
        data = self._read_data()
        users = data["users"]

        user.id = self._next_id(users)
        users.append(self._to_dict(user))

        self._write_data(data)
        return user

    def find_by_email(self, email: str) -> Optional[User]:
        data = self._read_data()
        for raw_user in data["users"]:
            if raw_user["email"] == email:
                return self._to_entity(raw_user)
        return None

    def find_by_id(self, user_id: int) -> Optional[User]:
        data = self._read_data()
        for raw_user in data["users"]:
            if raw_user["id"] == user_id:
                return self._to_entity(raw_user)
        return None

    def list_all(self) -> List[User]:
        data = self._read_data()
        return [self._to_entity(user) for user in data["users"]]
