from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager
import logging
import sys
from backend.settings import SHOW_STARTUP_MESSAGE, DOCS_URL, SWAGGER_URL, JWT_ACCESS_TOKEN_EXPIRES, JWT_SECRET_KEY
from backend.presentation.routes.default_routes import default_bp
from backend.presentation.routes.user_routes import user_bp
from backend.presentation.routes.swagger_routes import swagger_bp
from backend.presentation.routes.auth_routes import auth_bp
from backend.presentation.routes.model_routes import model_bp

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


load_dotenv()

def create_app():
    app = Flask(__name__)

    app.logger.setLevel("INFO")

    CORS(
        app,
        resources={r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "Accept",
                "X-Model-Variant",
            ],
            "expose_headers": ["Authorization"]
        }},
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Accept",
            "X-Model-Variant",
        ],
        expose_headers=["Authorization"],
        supports_credentials=False,
    )

    app.register_blueprint(default_bp)
    app.register_blueprint(user_bp, url_prefix="/users")
    app.register_blueprint(swagger_bp, url_prefix=SWAGGER_URL)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(model_bp, url_prefix="/model")

    # ✅ Startup message (modern Flask way)
    if SHOW_STARTUP_MESSAGE and DOCS_URL:
        app.logger.info("OpenAPI docs available at %s", DOCS_URL)

    app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = JWT_ACCESS_TOKEN_EXPIRES

    app.config["JWT_TOKEN_LOCATION"] = ["headers"]
    app.config["JWT_HEADER_NAME"] = "Authorization"
    app.config["JWT_HEADER_TYPE"] = "Bearer"

    # 🔥 CRITICAL
    app.config["JWT_COOKIE_CSRF_PROTECT"] = False
    app.config["JWT_CSRF_CHECK_FORM"] = False

    jwt = JWTManager(app)
    
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)