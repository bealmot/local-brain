#!/usr/bin/env python3
"""
router.py

Small HTTP API/router on top of your existing RAG + LM Studio pipeline.

Endpoints (JSON):

- GET  /health
- POST /chat
- POST /ingest
- POST /profile/regenerate
- GET  /profile
"""

from pathlib import Path
import subprocess
from typing import Optional, List
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Reuse your existing helpers from llm_rag_cli
from llm_rag_cli import (
    load_config,
    get_collection,
    retrieve_context,
    call_lm_studio,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PROFILE_PATH = HERE / "profile.md"
LOG_PATH = DATA_DIR / "conversations.jsonl"


app = FastAPI(title="Local RAG + LM Studio Router", version="0.1.0")


class ChatRequest(BaseModel):
    prompt: str
    use_rag: bool = True
    model: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    used_rag: bool
    model: str


def run_script(script_name: str) -> str:
    """
    Run a Python script in this project using the same interpreter as the server.
    Captures stdout and stderr and returns combined text.
    """
    cmd = ["python", str(HERE / script_name)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(HERE),
            capture_output=True,
            text=True,
            check=False,
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        combined = ""
        if out:
            combined += out
        if err:
            if combined:
                combined += "\n\n"
            combined += f"[stderr]\n{err}"
        return combined or "(no output)"
    except Exception as e:
        return f"[ERROR] Failed to run {script_name}: {e!r}"


@app.get("/health")
def health():
    """
    Very simple health check: verify config loads and index dir exists.
    (For a deeper health check, you can add an LM Studio ping.)
    """
    try:
        cfg = load_config()
        index_dir = Path(cfg["index_dir"]).expanduser()
        ok = index_dir.exists()
        return {
            "status": "ok" if ok else "warn",
            "index_dir": str(index_dir),
            "index_dir_exists": ok,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e!r}")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chat endpoint. Optionally uses RAG.
    Intended for API callers (e.g. n8n, scripts).
    """
    cfg = load_config()
    model_name = req.model or cfg["lm_studio"]["model"]

    used_rag_flag = False
    final_user_message = req.prompt
    rag_context_text = ""

    if req.use_rag:
        collection = get_collection(cfg)
        context = retrieve_context(collection, req.prompt, cfg)
        if context:
            used_rag_flag = True
            rag_context_text = context
            final_user_message = (
                "Use the following retrieved context to answer the question if it is relevant.\n"
                "If it is not relevant, ignore it and answer normally.\n\n"
                "### Retrieved context\n"
                f"{context}\n\n"
                "### Question\n"
                f"{req.prompt}"
            )

    reply = call_lm_studio(
        cfg,
        final_user_message,
        system_prompt=(
            "You are my local AI assistant running behind a router API.\n"
            "Use retrieved context when it clearly helps; otherwise rely on general reasoning.\n"
            "Be concise, technically accurate, and directly actionable.\n"
        ),
        model=model_name,
    )

    # Log as a web interaction (source='web'), since this is typically used by UIs / external tools
    log_interaction_web(
        user_prompt=req.prompt,
        sent_prompt=final_user_message,
        assistant_reply=reply,
        used_rag=used_rag_flag,
        model_name=model_name,
    )

    # Patch in rag_context for the log record if needed
    if rag_context_text:
        # Quick append; future ingestion only cares about user/assistant fields.
        try:
            # This is a bit hacky; if you want, we can make log_interaction_web accept rag_context too.
            pass
        except Exception:
            pass

    return ChatResponse(reply=reply, used_rag=used_rag_flag, model=model_name)

class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


@app.post("/v1/chat/completions")
def openai_chat_completions(req: OpenAIChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint for Open WebUI.

    - Uses all user messages concatenated as the 'prompt'
    - Uses RAG by default
    """
    cfg = load_config()
    model_name = req.model or cfg["lm_studio"]["model"]

    # Build a single prompt text from the conversation
    user_parts = []
    system_parts = []

    for m in req.messages:
        if m.role == "user":
            user_parts.append(m.content)
        elif m.role == "system":
            system_parts.append(m.content)

    user_prompt = "\n\n".join(user_parts).strip()
    system_prompt_extra = "\n\n".join(system_parts).strip()

    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user content found in messages")

    used_rag_flag = False
    final_user_message = user_prompt
    rag_context_text = ""

    # Always use RAG here; we want Open WebUI chats to use the brain.
    collection = get_collection(cfg)
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

    base_system_prompt = (
        "You are my local AI assistant accessed via Open WebUI.\n"
        "You have access to a RAG memory of my past conversations and notes.\n"
        "Use retrieved context when clearly helpful; otherwise answer normally.\n"
        "Be concise, technically accurate, and directly actionable.\n"
    )
    full_system_prompt = base_system_prompt
    if system_prompt_extra:
        full_system_prompt += "\n\nAdditional instructions from the UI:\n" + system_prompt_extra

    reply = call_lm_studio(
        cfg,
        final_user_message,
        system_prompt=full_system_prompt,
        model=model_name,
    )

    # Log the interaction
    log_interaction_web(
        user_prompt=user_prompt,
        sent_prompt=final_user_message,
        assistant_reply=reply,
        used_rag=used_rag_flag,
        model_name=model_name,
    )

    # Minimal OpenAI-style response
    import time
    resp = {
        "id": f"chatcmpl-router-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            # We don't compute real token counts here; Open WebUI usually doesn't require them.
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return resp

def log_interaction_web(
    user_prompt: str,
    sent_prompt: str,
    assistant_reply: str,
    used_rag: bool,
    model_name: str,
):
    """Append a web (OpenWebUI) interaction to data/conversations.jsonl."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "web",
            "model": model_name,
            "used_rag": used_rag,
            "user_prompt": user_prompt,
            "sent_prompt": sent_prompt,
            "rag_context": "",  # we'll fill this in if we have explicit context
            "assistant_reply": assistant_reply,
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Logging must never break the main flow
        print(f"[WARN] Failed to log web interaction: {e}", flush=True)



@app.post("/ingest")
def ingest():
    """
    Run ingest_conversations.py to add new logged CLI conversations into RAG.
    """
    output = run_script("ingest_conversations.py")
    return {"status": "ok", "output": output}


@app.post("/profile/regenerate")
def profile_regenerate():
    """
    Regenerate profile.md from current RAG state.
    """
    output = run_script("generate_profile.py")
    return {"status": "ok", "output": output}


@app.get("/profile")
def profile_get():
    """
    Return the current profile.md content.
    """
    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=404, detail="profile.md not found")
    content = PROFILE_PATH.read_text(encoding="utf-8")
    return {"path": str(PROFILE_PATH), "content": content}
