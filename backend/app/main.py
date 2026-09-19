"""FastAPI entry point: `uvicorn app.main:app --reload --port 8000` from backend/."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="MemeticWorld", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

logging.getLogger("memeticworld").info(
    "LLM provider=%s model=%s embeddings=%s db=%s", settings.llm_provider, settings.model_label,
    settings.embed_provider, settings.db_path)
