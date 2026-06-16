class DomainError(Exception):
    """Base class for domain-level errors"""
    pass

class InvalidCredentials(DomainError):
    pass


class UserAlreadyExists(DomainError):
    pass

class InvalidUserData(DomainError):
    pass