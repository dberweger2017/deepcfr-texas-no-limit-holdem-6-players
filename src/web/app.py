"""FastAPI application for the web poker client."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.utils.checkpoints import find_checkpoint_dirs
from src.utils.device import resolve_training_device
from src.web.session import SessionStore

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / "web" / "static"
store = SessionStore()


class CreateSessionRequest(BaseModel):
    models_dir: Optional[str] = "models/standard/phase1"
    hero_seat: int = Field(default=0, ge=0, le=5)
    stake: float = 200.0
    sb: float = 1.0
    bb: float = 2.0
    device: str = "cpu"


class ActionRequest(BaseModel):
    action: str
    amount: Optional[float] = None


def create_app() -> FastAPI:
    app = FastAPI(title="Rivermind Poker", version="1.0.0")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/model-dirs")
    def model_dirs() -> dict:
        return {
            "model_dirs": find_checkpoint_dirs(
                REPO_ROOT / "models",
                relative_to=REPO_ROOT,
            )
        }

    @app.post("/api/sessions")
    def create_session(body: CreateSessionRequest) -> dict:
        models_path = None
        if body.models_dir:
            candidate = REPO_ROOT / body.models_dir
            if candidate.is_dir():
                models_path = str(candidate)
        device = resolve_training_device(body.device)
        session = store.create(
            models_dir=models_path,
            hero_seat=body.hero_seat,
            stake=body.stake,
            sb=body.sb,
            bb=body.bb,
            device=device,
        )
        return session.serialize()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        try:
            return store.get(session_id).serialize()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/new-hand")
    def new_hand(session_id: str) -> dict:
        try:
            return store.get(session_id).start_new_hand()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/action")
    def hero_action(session_id: str, body: ActionRequest) -> dict:
        try:
            return store.get(session_id).apply_hero_action(body.action, body.amount)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/reveal")
    def reveal(session_id: str) -> dict:
        try:
            return store.get(session_id).toggle_reveal()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
