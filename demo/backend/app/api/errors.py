"""One error shape for every failure: {"error": {"code", "message"}}."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

JSON_BODY_LIMIT = 1024 * 1024
MULTIPART_OVERHEAD = 64 * 1024


class RequestBodyLimitMiddleware:
    """Limit incoming bytes before multipart parsing/spooling, even without a length header."""

    def __init__(self, app, max_upload_bytes: int):
        self.app = app
        self.upload_limit = max_upload_bytes + JSON_BODY_LIMIT + MULTIPART_OVERHEAD

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        upload = scope.get("path", "").rstrip("/") in ("/upload", "/api/upload")
        limit = self.upload_limit if upload else JSON_BODY_LIMIT
        lengths = [value for key, value in scope.get("headers", []) if key.lower() == b"content-length"]
        if lengths and (any(not value.isdigit() for value in lengths) or len(set(lengths)) != 1):
            return await JSONResponse(body("invalid_request", "invalid Content-Length"), status_code=400)(scope, receive, send)
        # Avoid converting an arbitrarily long attacker-supplied integer.
        normalized_length = (lengths[0].lstrip(b"0") or b"0") if lengths else None
        if normalized_length is not None and (len(normalized_length) > len(str(limit)) or int(normalized_length) > limit):
            return await JSONResponse(body("too_large", "request body exceeds the configured limit"), status_code=413)(scope, receive, send)
        total = 0
        exceeded = False
        replacement_sent = False

        async def reject():
            nonlocal replacement_sent
            if not replacement_sent:
                replacement_sent = True
                await JSONResponse(body("too_large", "request body exceeds the configured limit"), status_code=413)(scope, receive, send)

        async def limited_receive():
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    exceeded = True
                    # Starlette closes partially spooled uploads when this
                    # error interrupts form parsing. The wrapper preserves 413
                    # even if a framework layer replaces its error response.
                    raise MultiPartException("request body exceeds the configured limit")
            return message

        async def guarded_send(message):
            if exceeded:
                await reject()
            else:
                await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
            await reject()


class ApiError(Exception):
    """Raise from an endpoint to return a coded error."""

    def __init__(self, status: int, code: str, message: str) -> None:
        """Store status, machine code and human message."""
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def body(code: str, message: str) -> dict:
    """The JSON error body."""
    return {"error": {"code": code, "message": message}}


def install(app: FastAPI, settings=None) -> None:
    """Register handlers that turn every error into the standard shape."""
    app.add_middleware(RequestBodyLimitMiddleware,
                       max_upload_bytes=settings.max_upload_bytes if settings is not None else 500 * 1024 * 1024)

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
