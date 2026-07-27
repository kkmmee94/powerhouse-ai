class PowerhouseError(Exception):
    """Base application error."""


class ModelUnavailableError(PowerhouseError):
    """Raised when the local inference server cannot be reached."""


class ModelResponseError(PowerhouseError):
    """Raised when the local model returns an unusable response."""


class ResourceBudgetError(PowerhouseError):
    """Raised when a task cannot be run inside the configured resource budget."""
