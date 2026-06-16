from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required

from backend.domain.exception import InvalidCredentials
from backend.di import user_service

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()

    try:
        user = user_service.authenticate(
            username=data["identifier"],
            password=data["password"]
        )

        access_token = create_access_token(
            identity=str(user.id),
            additional_claims={
                "full_name": user.full_name,
                "username": user.username,
                "email": user.email,
            }
        )

        return jsonify({
            "access_token": access_token,
            "user": {
                "id": user.id,
                "full_name": user.full_name,
                "username": user.username,
                "email": user.email,
            }
        }), 200

    except InvalidCredentials:
        return jsonify({"error": "Incorrect username or password"}), 401
    
@auth_bp.route("/token", methods=["HEAD"])
@jwt_required()
def validate_token():
    return "", 200
