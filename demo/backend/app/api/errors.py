"""One error shape for every failure: {"error": {"code", "message"}}."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    """Raise from an endpoint to return a coded error."""

    def __init__(self, status: int, code: str, message: str) -> None:
        """Store status, machine code and human message."""
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def body(code: str, message: str) -> dict:
    """The JSON error body."""
    return {"error": {"code": code, "message": message}}


def install(app: FastAPI) -> None:
    """Register handlers that turn every error into the standard shape."""

    @app.exception_handler(ApiError)
    async def _api(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(body("not_found" if exc.status_code == 404 else "http_error", str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"] if p != "body")
        return JSONResponse(body("invalid_request", f"{where}: {first['msg']}" if where else first["msg"]), status_code=422)
