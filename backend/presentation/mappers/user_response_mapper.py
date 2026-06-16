from backend.domain.entities.user import User

class UserResponseMapper:
    @staticmethod
    def to_json(user: User) -> dict:
        return {
            "id": user.id,
            "full_name": user.full_name,
            "username": user.username,
            "email": user.email,
            "avatar_color": user.avatar_color,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }