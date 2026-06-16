from flask import Blueprint, jsonify

from backend.di import health_service

default_bp = Blueprint("default", __name__)

@default_bp.route("/health/app", methods=["HEAD"])
def health():
    return "", 200

@default_bp.route("/health/database", methods=["GET"])
def ping():
    try:
        health_service.check()

        return jsonify({
            "database": "postgresql",
            "provider": "supabase",
            "status": "healthy"
        }), 200

    except Exception as e:
        return jsonify({
            "database": "postgresql",
            "provider": "supabase",
            "status": "unhealthy",
            "error": str(e)
        }), 500


@default_bp.route("/teapot", methods=["GET"])
def teapot():
    return jsonify({
        "message": "I'm a teapot ☕",
    }), 418
