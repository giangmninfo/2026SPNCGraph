from flask import Blueprint, request, jsonify
from werkzeug.security import check_password_hash

from backend.domain.entities.user import User
from backend.domain.exception import InvalidCredentials, InvalidUserData
from backend.presentation.mappers.user_response_mapper import UserResponseMapper
from backend.di import user_service

user_bp = Blueprint("users", __name__, url_prefix="/users")

@user_bp.route("", methods=["GET"])
def list_users():
    users = user_service.list_users()

    return jsonify({
        "status": True,
        "data": [UserResponseMapper.to_json(u) for u in users]
    }), 200

@user_bp.route("", methods=["POST"])
def create_user():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        user = User(
            id=None,
            full_name=data.get("full_name"),
            username=data.get("username"),
            email=data.get("email"),
            password=data.get("password"),
            avatar_color=data.get("avatar_color"),
            created_at=None,
        )

        created = user_service.create_user(user)

        return jsonify({
            "id": created.id,
            "username": created.username,
            "email": created.email,
        }), 201

    except (ValueError, InvalidUserData) as e:
        return jsonify({"error": str(e)}), 400

@user_bp.route("", methods=["DELETE"])
def delete_user():
    """
    Dummy endpoint – user deletion not implemented yet
    """
    return jsonify({
        "status": False,
        "message": "DELETE /users is not implemented yet"
    }), 501

@user_bp.route("", methods=["PATCH"])
def update_user():
    """
    Dummy endpoint – user update not implemented yet
    """
    return jsonify({
        "status": False,
        "message": "PATCH /users is not implemented yet"
    }), 501

@user_bp.route("", methods=["OPTIONS"])
def users_options():
    """
    OPTIONS endpoint for /users
    """
    return jsonify({
        "methods": ["GET", "POST", "PATCH", "DELETE", "OPTIONS"]
    }), 200

@user_bp.route("/availability", methods=["GET"])
def username_availability():
    username = request.args.get("username")

    if not username:
        return jsonify({"error": "username is required"}), 400

    available = user_service.is_username_available(username)

    return jsonify({
        "available": available
    })

# @user_bp.route("/test-create", methods=["GET"])
# def test_create():
#     # Generate unique credentials every time
#     suffix = uuid.uuid4().hex[:8]
#     username = f"route_user_{suffix}"
#     email = f"{username}@example.com"

#     user = User(
#         id=None,
#         full_name="Route User",
#         username=username,
#         email=email,
#         password=b"123",          # BYTEA → OK
#         avatar_color="#00ffaa",
#         created_at=None,
#     )

#     created = user_service.create_user(user)

#     return jsonify({
#         "id": created.id,
#         "username": created.username,
#         "email": created.email,
#     }), 201