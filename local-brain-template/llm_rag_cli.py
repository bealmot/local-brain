#!/usr/bin/env python3
import argparse
import pathlib
import sys
import textwrap
import json
from datetime import datetime, timezone

import yaml
import chromadb
from chromadb.config import Settings
import requests


HERE = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
DATA_DIR = HERE / "data"
LOG_PATH = DATA_DIR / "conversations.jsonl"

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_collection(cfg: dict):
    index_dir = pathlib.Path(cfg["index_dir"]).expanduser()
    collection_name = cfg["rag"]["collection_name"]
    client = chromadb.PersistentClient(
        path=str(index_dir),
        settings=Settings(allow_reset=False),
    )
    return client.get_or_create_collection(name=collection_name)


def retrieve_context(collection, query: str, cfg: dict) -> str:
    top_k = int(cfg["rag"].get("top_k", 8))
    max_chars = int(cfg["rag"].get("max_context_chars", 8000))

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    context_chunks = []
    total_chars = 0

    for doc, meta in zip(docs, metas):
        piece = f"[{meta.get('conversation_title', 'unknown')}] ({meta.get('role')}):\n{doc}\n"
        if total_chars + len(piece) > max_chars:
            break
        context_chunks.append(piece)
        total_chars += len(piece)

    if not context_chunks:
        return ""

    header = (
        "The following context is retrieved from my past ChatGPT conversations. "
        "Use it to ground your answer, but do NOT repeat it verbatim unless necessary.\n\n"
    )
    return header + "\n\n".join(context_chunks)


def call_lm_studio(cfg: dict, user_message: str, system_prompt: str = "", model: str | None = None) -> str:
    lm_cfg = cfg["lm_studio"]
    base_url = lm_cfg["base_url"].rstrip("/")
    api_key = lm_cfg["api_key"]
    model = model or lm_cfg["model"]

    url = f"{base_url}/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] Unexpected response from LM Studio: {e}\nRaw: {data}"

def log_interaction(
    user_prompt: str,
    final_user_message: str,
    assistant_reply: str,
    used_rag: bool,
    rag_context: str | None,
    cfg: dict,
):
    """Append a single interaction to data/conversations.jsonl."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        lm_cfg = cfg.get("lm_studio", {})
        model = lm_cfg.get("model", "unknown")

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "cli",  # later: 'web', 'agent', etc.
            "model": model,
            "used_rag": used_rag,
            # What *you* actually typed
            "user_prompt": user_prompt,
            # The full prompt actually sent to the model (with wrappers/context)
            "sent_prompt": final_user_message,
            # The retrieved context as its own field (if any)
            "rag_context": rag_context or "",
            # The model's reply
            "assistant_reply": assistant_reply,
        }

        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as e:
        # Logging should never crash your main workflow.
        print(f"[WARN] Failed to log interaction: {e}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RAG + LM Studio CLI helper"
    )
    parser.add_argument(
        "prompt",
        nargs="+",
        help="Your question / instruction for the model."
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable retrieval and just send the prompt to LM Studio."
    )
    parser.add_argument(
        "--model",
        help="Override model name from config.yaml."
    )
    parser.add_argument(
        "--system",
        help="Optional system prompt file to prepend.",
    )

    args = parser.parse_args(argv)
    user_prompt = " ".join(args.prompt)

    cfg = load_config()

    system_parts = [
        "You are my local AI assistant. Your goal is to help me solve problems efficiently.",
        "You have access to a retrieval index of my past ChatGPT conversations and personal technical notes.",
        "Use retrieved context when it is clearly relevant and improves precision, otherwise rely on general reasoning.",
        "Keep answers concise, technically accurate, and directly actionable.",
        "If the query depends on missing details, ask for the minimal clarification needed.",
        "Avoid hallucinating; if uncertain, state the uncertainty and suggest a safe next action.",
    ]

    if args.system:
        path = pathlib.Path(args.system).expanduser()
        if path.exists():
            system_parts.append(path.read_text(encoding='utf-8'))
        else:
            print(f"[WARN] System prompt file not found: {path}", file=sys.stderr)


    system_prompt = "\n\n".join(system_parts)

    used_rag_flag = False          # did we actually include retrieved context?
    rag_context_text = ""          # store context separately for logging

    if args.no_rag:
        final_user_message = user_prompt
    else:
        collection = get_collection(cfg)
        context = retrieve_context(collection, user_prompt, cfg)
        if context:
            used_rag_flag = True
            rag_context_text = context

            final_user_message = textwrap.dedent(
                f"""\
                Use the following retrieved context to answer the question if it is relevant.
                If it is not relevant, ignore it and answer normally.

                ### Retrieved context
                {context}

                ### Question
                {user_prompt}
                """
            ).strip()
        else:
            final_user_message = user_prompt


    # Call the model
    answer = call_lm_studio(
        cfg,
        final_user_message,
        system_prompt=system_prompt,
        model=args.model,
    )

    # Print to stdout for you
    print(answer)

    # Log the interaction for future ingestion
    log_interaction(
        user_prompt=user_prompt,
        final_user_message=final_user_message,
        assistant_reply=answer,
        used_rag=used_rag_flag,
        rag_context=rag_context_text,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
