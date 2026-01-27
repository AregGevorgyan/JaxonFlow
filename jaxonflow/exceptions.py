"""Custom exceptions for JaxonFlow."""

from typing import Any


class JaxonFlowError(Exception):
    """Base exception for all JaxonFlow errors."""

    pass


class ConfigurationError(JaxonFlowError):
    """Raised when there's an error in configuration."""

    pass


class HardwareDetectionError(JaxonFlowError):
    """Raised when hardware detection fails."""

    pass


class KernelGenerationError(JaxonFlowError):
    """Raised when kernel generation fails."""

    def __init__(self, message: str, operation: str | None = None, iterations: int = 0) -> None:
        self.operation = operation
        self.iterations = iterations
        super().__init__(message)


class KernelCompilationError(JaxonFlowError):
    """Raised when kernel compilation fails."""

    def __init__(self, message: str, code: str | None = None, error_details: str | None = None) -> None:
        self.code = code
        self.error_details = error_details
        super().__init__(message)


class KernelVerificationError(JaxonFlowError):
    """Raised when kernel verification fails."""

    def __init__(
        self,
        message: str,
        max_abs_error: float | None = None,
        max_rel_error: float | None = None,
        test_case: str | None = None,
    ) -> None:
        self.max_abs_error = max_abs_error
        self.max_rel_error = max_rel_error
        self.test_case = test_case
        super().__init__(message)


class CacheError(JaxonFlowError):
    """Raised when cache operations fail."""

    pass


class LLMError(JaxonFlowError):
    """Base exception for LLM-related errors."""

    pass


class LLMAPIError(LLMError):
    """Raised when LLM API calls fail."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        response: Any | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class LLMResponseParseError(LLMError):
    """Raised when LLM response cannot be parsed."""

    def __init__(self, message: str, raw_response: str | None = None) -> None:
        self.raw_response = raw_response
        super().__init__(message)


class AgentError(JaxonFlowError):
    """Base exception for agent-related errors."""

    pass


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""

    def __init__(self, message: str, agent_role: str | None = None, timeout_seconds: float | None = None) -> None:
        self.agent_role = agent_role
        self.timeout_seconds = timeout_seconds
        super().__init__(message)


class FallbackError(JaxonFlowError):
    """Raised when fallback to XLA/Inductor also fails."""

    pass
