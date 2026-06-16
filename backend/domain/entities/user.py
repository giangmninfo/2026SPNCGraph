from datetime import datetime
from typing import Optional


class User:
    def __init__(
        self,
        id: Optional[int],
        full_name: str,
        username: str,
        email: str,
        password: bytes,
        avatar_color: str,
        created_at: Optional[datetime] = None,
    ):
        # Domain validations
        if not full_name:
            raise ValueError("Full name cannot be empty")

        if not username:
            raise ValueError("Username cannot be empty")

        if "@" not in email:
            raise ValueError("Invalid email address")

        if not password:
            raise ValueError("Password cannot be empty")

        if not avatar_color or not avatar_color.startswith("#"):
            raise ValueError("Invalid avatar color")

        self.id = id
        self.full_name = full_name
        self.username = username
        self.email = email
        self.password = password
        self.avatar_color = avatar_color
        self.created_at = created_at
