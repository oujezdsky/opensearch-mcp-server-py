class OpenSearchMCPError(Exception):
    """Base exception for all errors in the OpenSearch MCP project."""

    pass


class HelperOperationError(OpenSearchMCPError):
    """Error raised when a helper function fails to perform its operation.
    Enriches the exception with runtime context from the helper call.
    """

    def __init__(self, *, message: str, func_name: str, action: str, original: Exception):
        """Initialize the HelperOperationError.

        Args:
            message: A high-level message for the exception (rarely used directly).
            func_name: The name of the function where the error occurred.
            action: A description of the action that failed (e.g., "get agentic memory").
            original: The original exception that was caught and wrapped.
        """
        super().__init__(message)
        self.func_name = func_name
        self.action = action
        self.original = original

    def __str__(self):
        """Return a human-readable string representation of the error."""
        return f'{self.func_name} failed to {self.action}: {self.original}'
