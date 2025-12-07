#!/usr/bin/env python3
"""
brain_api.py

v1 "brain service" for your local setup.

- Exposes OpenAI-compatible endpoints:
    - GET  /v1/models
    - POST /v1/chat/completions
- Also exposes:
    - GET  /health
    - POST /chat     (simple CLI/testing endpoint)

- Uses your existing RAG index (Chroma) + LM Studio backend
- Logs all interactions to data/conversations.jsonl

Frontends (Open WebUI, CLI, n8n, etc.) should talk to THIS,
not directly to LM Studio.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import json
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Reuse your existing helpers from llm_rag_cli.py
from llm_rag_cli import (
    load_config,
    get_collection,
    retrieve_context,
    call_lm_studio,
)

# -------------------------------------------------------------------
# Paths & basic setup
# -------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
LOG_PATH = DATA_DIR / "conversations.jsonl"

app = FastAPI(
    title="Local Brain API",
    version="0.1.0",
    description="RAG + LM Studio backed personal brain.",
)

# -------------------------------------------------------------------
# Pydantic models for OpenAI-style API
# -------------------------------------------------------------------

class OpenAIChatMessage(BaseModel):
    role: str = Field(..., description="system | user | assistant")
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(
        None,
        description="Model name. If omitted, uses lm_studio.model from config.yaml",
    )
    messages: List[OpenAIChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False  # streaming not implemented

# Simple /chat wrapper
class SimpleChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

def log_interaction(
    source: str,
    model_name: str,
    used_rag: bool,
    user_prompt: str,
    sent_prompt: str,
    rag_context: str,
    assistant_reply: str,
) -> None:
    """
    Append a JSONL record to data/conversations.jsonl.
    This will be the canonical log for later ingestion.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,  # e.g. "web", "cli", "openwebui"
            "model": model_name,
            "used_rag": used_rag,
            "user_prompt": user_prompt,
            "sent_prompt": sent_prompt,
            "rag_context": rag_context,
            "assistant_reply": assistant_reply,
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Logging must NEVER break the main flow
        print(f"[WARN] Failed to log interaction: {e}", flush=True)

# -------------------------------------------------------------------
# Core RAG + LM Studio logic
# -------------------------------------------------------------------

def run_rag_completion(
    req: OpenAIChatCompletionRequest,
    source: str = "web",
) -> dict:
    """
    Core logic: given an OpenAI-style request, run RAG + LM Studio and
    return an OpenAI-style response dict.
    """
    cfg = load_config()
    model_name = req.model or cfg["lm_studio"]["model"]

    # Extract user + system content
    user_parts: List[str] = []
    system_parts: List[str] = []

    for m in req.messages:
        if m.role == "user":
            user_parts.append(m.content)
        elif m.role == "system":
            system_parts.append(m.content)

    user_prompt = "\n\n".join(user_parts).strip()
    system_extra = "\n\n".join(system_parts).strip()

    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user content found in messages")

    # RAG retrieval
    used_rag_flag = False
    rag_context_text = ""
    final_user_message = user_prompt

    collection = get_collection(cfg)
    if collection is not None:
        context = retrieve_context(collection, user_prompt, cfg)
        if context:
            used_rag_flag = True
            rag_context_text = context
            final_user_message = (
                "Use the following retrieved context to answer the question if it is relevant.\n"
                "If it is not relevant, ignore it and answer normally.\n\n"
                "### Retrieved context\n"
                f"{context}\n\n"
                "### Question\n"
                f"{user_prompt}"
            )

    # System prompt
    base_system_prompt = (
        "You are my local AI assistant, backed by a personal memory index (RAG).\n"
        "Use retrieved context when it clearly helps; otherwise answer normally.\n"
        "Be concise, technically accurate, and directly actionable.\n"
    )
    if system_extra:
        base_system_prompt += "\n\nAdditional UI/system instructions:\n" + system_extra

    # ----------------------------------------------------------------
    # Call LM Studio
    # IMPORTANT: do NOT pass unsupported kwargs like temperature/max_tokens
    # ----------------------------------------------------------------
    answer = call_lm_studio(
        cfg,
        final_user_message,
        system_prompt=base_system_prompt,
        model=model_name,
    )

    # Log interaction
    log_interaction(
        source=source,
        model_name=model_name,
        used_rag=used_rag_flag,
        user_prompt=user_prompt,
        sent_prompt=final_user_message,
        rag_context=rag_context_text,
        assistant_reply=answer,
    )

    # Minimal OpenAI-style response
    now = int(time.time())
    resp = {
        "id": f"chatcmpl-localbrain-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            # If you later wire token counting, update these
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return resp

# -------------------------------------------------------------------
# FastAPI endpoints
# -------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Basic health check:
    - Can we load config?
    - Does the index dir exist?
    """
    try:
        cfg = load_config()
        index_dir = Path(cfg["index_dir"]).expanduser()
        lm_cfg = cfg["lm_studio"]
        return {
            "status": "ok",
            "index_dir": str(index_dir),
            "index_dir_exists": index_dir.exists(),
            "lm_studio": {
                "base_url": lm_cfg.get("base_url", ""),
                "model": lm_cfg.get("model", ""),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e!r}")

# -------- OpenAI-compatible endpoints --------

@app.get("/v1/models")
def list_models():
    """
    Minimal OpenAI-compatible model list.
    Open WebUI uses this to discover available models.
    """
    cfg = load_config()
    model_name = cfg["lm_studio"]["model"]

    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "local-brain",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: OpenAIChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    This is what Open WebUI (and any other OpenAI client) should call.
    """
    return run_rag_completion(req, source="web")

# -------- Simple /chat for CLI/testing --------

@app.post("/chat")
def chat_simple(req: SimpleChatRequest):
    """
    Simpler /chat endpoint for CLI, scripts, etc.
    """
    oai_req = OpenAIChatCompletionRequest(
        model=req.model,
        messages=[OpenAIChatMessage(role="user", content=req.prompt)],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    resp = run_rag_completion(oai_req, source="cli")
    return {"reply": resp["choices"][0]["message"]["content"], "raw": resp}
