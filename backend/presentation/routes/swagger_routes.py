from backend.settings import SWAGGER_URL, API_URL
from flask_swagger_ui import get_swaggerui_blueprint # type: ignore

swagger_bp = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={"app_name": "GNN Classifier API"})