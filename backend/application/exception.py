class ApplicationError(Exception):
    """Base class for application-level errors"""
    pass

class MLServiceUnavailable(ApplicationError):
    pass