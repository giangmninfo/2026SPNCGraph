from sqlalchemy import text
from backend.infrastructure.database.postgres import SessionLocal
from backend.infrastructure.database.postgres import engine
from backend.interfaces.repositories.health_repository_interface import IHealthRepository


class PostgresHealthRepository(IHealthRepository):
    def check(self) -> None:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))