from backend.interfaces.services.health_service_interface import IHealthService
from backend.infrastructure.repositories.postgres_health_repository import PostgresHealthRepository

class HealthService(IHealthService):
    def __init__(self, repository=PostgresHealthRepository()):
        self.repository = repository

    def check(self) -> None:
        self.repository.check()
