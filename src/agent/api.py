"""The HTTP API and the demo page.

    POST /v1/ask          ask a question about a contract (needs an API key; optional Idempotency-Key header)
    GET  /v1/contracts    find a contract's exact title
    GET  /v1/usage        what this key has spent today
    POST /v1/feedback     thumbs up or down for a conversation
    GET  /v1/notes        notes the model proposed and the ones you approved
    POST /v1/notes/{id}/approve   let the model see a note in later questions
    DELETE /v1/notes/{id} remove a note
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
from agent.embeddings import DEFAULT_THRESHOLDS, make_embedder
from agent.config import Settings, get_settings
from agent.idempotency import MAX_KEY_CHARS, IdempotencyStore, fingerprint
from agent.llm import RouterLLM
from agent.memory import Memory
from agent.service import AgentService, BadRequest
from gateway.service import GatewayService

log = logging.getLogger("agent.api")
STATIC = Path(__file__).parent / "static"


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    helpful: bool


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    contract: str | None = Field(default=None, max_length=300)
    session_id: str | None = Field(default=None, max_length=64)


def build_service(settings: Settings | None = None) -> AgentService:
    """The real service: real models, real databases under AGENT_DATA_DIR."""
    settings = settings or get_settings()
    data = Path(os.environ.get("AGENT_DATA_DIR", "./data/agent"))
    data.mkdir(parents=True, exist_ok=True)

    kind = os.environ.get("AGENT_CACHE_EMBEDDER", "openai")
    accounts = Accounts(
        data / "accounts.db",
        global_daily_budget_usd=float(os.environ.get("AGENT_GLOBAL_DAILY_BUDGET_USD", "5.0")),
    )
    # one cache file per embedder, because vectors from different models cannot be compared
    cache = SemanticCache(
        make_embedder(kind), data / f"cache_{kind}.db",
        threshold=float(os.environ.get("AGENT_CACHE_THRESHOLD", DEFAULT_THRESHOLDS[kind])),
    )
    challenger = None
    if settings.ab_model and settings.ab_percent:
        challenger = RouterLLM(settings, models=[settings.ab_model, *settings.fallback_models])
    return AgentService(
        settings, RouterLLM(settings), GatewayService(), Memory(data / "memory.db"), accounts, cache,
        max_concurrent=int(os.environ.get("AGENT_MAX_CONCURRENT_RUNS", "4")), challenger=challenger,
        idempotency=IdempotencyStore(data / "idempotency.db"),
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
    async def ask(body: AskRequest, request: Request, key: ApiKey = Depends(authenticated)):
        service = svc(request)
        idem = request.headers.get("Idempotency-Key")
        if idem is None:
            return _reply(await service.ask(key, body.question, body.contract, body.session_id))
        if not idem or len(idem) > MAX_KEY_CHARS:
            raise HTTPException(422, f"Idempotency-Key must be 1 to {MAX_KEY_CHARS} characters")

        request_hash = fingerprint(question=body.question.strip(), contract=body.contract, session_id=body.session_id)
        begin = service.idempotency.begin(key.id, idem, request_hash)
        if begin.kind == "replay":
            return JSONResponse(begin.body, status_code=begin.status, headers={"Idempotent-Replay": "true"})
        if begin.kind == "mismatch":
            raise HTTPException(422, "this Idempotency-Key was already used for a different request")
        if begin.kind == "in_flight":
            raise HTTPException(409, "a request with this Idempotency-Key is still running")
        try:
            response = _reply(await service.ask(key, body.question, body.contract, body.session_id))
        except BaseException:
            service.idempotency.abandon(key.id, idem)
            raise
        if isinstance(response, JSONResponse):  # a 503 is not stored, so the retry runs again
            service.idempotency.abandon(key.id, idem)
        else:
            service.idempotency.finish(key.id, idem, request_hash, 200, response)
        return response

    @app.post("/v1/feedback")
    async def feedback(body: FeedbackRequest, request: Request, key: ApiKey = Depends(authenticated)):
        arm = svc(request).memory.add_feedback(body.session_id, key.id, body.helpful)
        if arm is None:
            raise HTTPException(404, "no such session")
        metrics.FEEDBACK.labels(arm, str(body.helpful).lower()).inc()
        return {"recorded": True}

    @app.get("/v1/notes")
    async def notes(request: Request, matter: str | None = None, key: ApiKey = Depends(authenticated)):
        return {"notes": svc(request).memory.list_notes(key.id, matter)}

    @app.post("/v1/notes/{note_id}/approve")
    async def approve_note(note_id: int, request: Request, key: ApiKey = Depends(authenticated)):
        if not svc(request).memory.approve_note(key.id, note_id):
            raise HTTPException(404, "no such note")
        return {"approved": True}

    @app.delete("/v1/notes/{note_id}")
    async def delete_note(note_id: int, request: Request, key: ApiKey = Depends(authenticated)):
        if not svc(request).memory.delete_note(key.id, note_id):
            raise HTTPException(404, "no such note")
        return {"deleted": True}

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
        svc(request).idempotency.delete_for_key(key.id)
        return {"deleted": True}

    return app


def _reply(answer):
    """The HTTP body for an answer."""
    if answer.stop_reason in ("error", "timeout"):
        # the model service is down. A 200 would look like a normal answer to a client that only checks the status.
        return JSONResponse(
            {"error": answer.answer, "stop_reason": answer.stop_reason, "session_id": answer.session_id},
            status_code=503, headers={"Retry-After": "10"},
        )
    return answer.to_dict()


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
