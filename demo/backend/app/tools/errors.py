"""Typed errors raised by the tools package."""
from __future__ import annotations


class ToolError(Exception):
    """Base class for all tool errors, carrying a machine code and a human message."""

    default_code = "tool_error"

    def __init__(self, message: str, code: str | None = None) -> None:
        """Store the human message and an optional machine code."""
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code

    def __str__(self) -> str:
        """Return the code and message together."""
        return f"[{self.code}] {self.message}"


class UnsupportedFile(ToolError):
    """The file is missing, not a regular file, or not a supported image type."""

    default_code = "unsupported_file"


class TooLarge(ToolError):
    """The file or image exceeds a configured size limit."""

    default_code = "too_large"


class CorruptFile(ToolError):
    """The file is empty, truncated or otherwise undecodable."""

    default_code = "corrupt_file"


class OcrUnavailable(ToolError):
    """No OCR engine is installed."""

    default_code = "ocr_unavailable"
