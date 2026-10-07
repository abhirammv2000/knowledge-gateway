"""The HTTP API and the demo page.

    POST /v1/ask          ask a question about a contract (needs an API key)
    GET  /v1/contracts    find a contract's exact title
    GET  /v1/usage        what this key has spent today
    DELETE /v1/sessions/{id}   forget one conversation
    DELETE /v1/data       forget everything stored for this key
    GET  /metrics         Prometheus metrics (admin key)
    GET  /healthz         is it up
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from agent import metrics
from agent.accounts import Accounts, ApiKey, Denied
from agent.cache import SemanticCache
from agent.config import Settings, get_settings
from agent.llm import RouterLLM
from agent.memory import Memory
from agent.service import AgentService, BadRequest
from agent.tools import model_lock
from gateway.service import GatewayService

log = logging.getLogger("agent.api")
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    contract: str | None = Field(default=None, max_length=300)
    session_id: str | None = Field(default=None, max_length=64)


def build_service(settings: Settings | None = None) -> AgentService:
    """The real service: real models, real databases under AGENT_DATA_DIR."""
    from gateway.retrieval import get_embedder

    settings = settings or get_settings()
    data = Path(os.environ.get("AGENT_DATA_DIR", "./data/agent"))
    data.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder()

    def _embed(text: str):
        with model_lock:
            return embedder.encode([text], normalize_embeddings=True)[0]

    accounts = Accounts(
        data / "accounts.db",
        global_daily_budget_usd=float(os.environ.get("AGENT_GLOBAL_DAILY_BUDGET_USD", "5.0")),
    )
    cache = SemanticCache(
        _embed, data / "cache.db",
        threshold=float(os.environ.get("AGENT_CACHE_THRESHOLD", "0.92")),
    )
    return AgentService(
        settings, RouterLLM(settings), GatewayService(), Memory(data / "memory.db"), accounts, cache,
        max_concurrent=int(os.environ.get("AGENT_MAX_CONCURRENT_RUNS", "4")),
    )


def create_app(service: AgentService | None = None, warm_up: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or build_service()
        if warm_up:
            await asyncio.to_thread(_warm_up, app.state.service)
        yield

    app = FastAPI(title="Contract review agent", version="1.0", lifespan=lifespan)
    if service is not None:
        app.state.service = service

    def svc(request: Request) -> AgentService:
        return request.app.state.service

    def authenticated(request: Request, authorization: str = Header(default="")) -> ApiKey:
        token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        key = svc(request).accounts.authenticate(token.strip())
        if key is None:
            raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return key

    def admin(key: ApiKey = Depends(authenticated)) -> ApiKey:
        if not key.is_admin:
            raise HTTPException(403, "this needs an admin key")
        return key

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(Denied)
    async def denied(_: Request, exc: Denied):
        metrics.DENIED.labels(str(exc.status)).inc()
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse({"error": exc.message}, status_code=exc.status, headers=headers)

    @app.exception_handler(BadRequest)
    async def bad_request(_: Request, exc: BadRequest):
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse({"error": "internal error"}, status_code=500)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    async def demo_page() -> str:
        return (STATIC / "demo.html").read_text(encoding="utf-8")

    @app.get("/metrics")
    async def prometheus(_: ApiKey = Depends(admin)) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/contracts")
    async def contracts(request: Request, contains: str = "", limit: int = 25, _: ApiKey = Depends(authenticated)):
        return await asyncio.to_thread(svc(request).gateway.list_contracts, contains, limit)

    @app.post("/v1/ask")
    async def ask(body: AskRequest, request: Request, key: ApiKey = Depends(authenticated)) -> dict[str, Any]:
        answer = await svc(request).ask(key, body.question, body.contract, body.session_id)
        return answer.to_dict()

    @app.get("/v1/usage")
    async def usage(request: Request, key: ApiKey = Depends(authenticated)) -> dict[str, Any]:
        spend = svc(request).accounts.spend(key)
        return {
            "key": key.name, "requests_today": spend.requests, "spent_today_usd": spend.spent_usd,
            "daily_budget_usd": spend.budget_usd, "requests_per_minute": key.rpm,
        }

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request, key: ApiKey = Depends(authenticated)):
        if not svc(request).memory.delete_session(session_id, key.id):
            raise HTTPException(404, "no such session")
        return {"deleted": True}

    @app.delete("/v1/data")
    async def delete_data(request: Request, key: ApiKey = Depends(authenticated)):
        svc(request).memory.delete_all(key.id)
        return {"deleted": True}

    return app


def _warm_up(service: AgentService) -> None:
    """Load the models now, so the first real question is not the one that waits for them."""
    from agent.guard import Redactor

    Redactor()("warm up")
    service.gateway.list_contracts()
    if service.cache is not None:
        service.cache._embed("warm up")


def app_factory() -> FastAPI:
    """What uvicorn runs: uvicorn agent.api:app_factory --factory"""
    return create_app(warm_up=True)
