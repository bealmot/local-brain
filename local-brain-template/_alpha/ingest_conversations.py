#!/usr/bin/env python3
"""
ingest_conversations.py

Ingest new CLI conversations from data/conversations.jsonl into the Chroma index,
so that future RAG queries can see your live interactions as part of memory.
"""

import json
from pathlib import Path

import yaml
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Reuse config logic from llm_rag_cli if you want, but keep this standalone too.
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
LOG_PATH = DATA_DIR / "conversations.jsonl"
STATE_PATH = DATA_DIR / "conversations_ingest_state.json"
CONFIG_PATH = HERE / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state() -> int:
    """
    Return last ingested line number (1-based).
    If no state file, assume nothing ingested yet (0).
    """
    if not STATE_PATH.exists():
        return 0
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return int(data.get("last_line", 0))
    except Exception:
        return 0


def save_state(last_line: int) -> None:
    STATE_PATH.write_text(
        json.dumps({"last_line": last_line}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_new_records(last_line: int):
    """
    Yield (line_no, record_dict) for new lines after last_line.
    """
    if not LOG_PATH.exists():
        print(f"[INFO] No conversation log found at {LOG_PATH}")
        return

    with LOG_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i <= last_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[WARN] Failed to parse JSON on line {i}: {e}")
                continue
            yield i, rec


def main():
    cfg = load_config()
    index_dir = Path(cfg["index_dir"]).expanduser()
    collection_name = cfg["rag"]["collection_name"]

    index_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(index_dir),
        settings=Settings(allow_reset=False),
    )

    collection = client.get_or_create_collection(name=collection_name)

    last_line = load_state()
    print(f"[INFO] Last ingested line: {last_line}")

    # Embedding model (same as rag_index.py)
    print("[INFO] Loading embedding model…")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    docs = []
    metadatas = []
    ids = []
    new_last_line = last_line

    for line_no, rec in iter_new_records(last_line):
        new_last_line = max(new_last_line, line_no)

        user_prompt = rec.get("user_prompt", "").strip()
        assistant_reply = rec.get("assistant_reply", "").strip()
        timestamp = rec.get("timestamp", "")
        model_name = rec.get("model", "")
        used_rag = bool(rec.get("used_rag", False))

        if not user_prompt and not assistant_reply:
            continue

        # Compose a document text from user + assistant
        text = (
            f"[{timestamp}] (cli, model={model_name}, used_rag={used_rag})\n"
            f"User:\n{user_prompt}\n\n"
            f"Assistant:\n{assistant_reply}\n"
        ).strip()

        docs.append(text)
        metadatas.append(
            {
                "source": "live_cli",
                "timestamp": timestamp,
                "model": model_name,
                "used_rag": used_rag,
            }
        )
        ids.append(f"live-cli-{line_no}")

        # Batch insert every 64 docs
        if len(docs) >= 64:
            embeddings = model.encode(docs, show_progress_bar=False).tolist()
            collection.add(
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            print(f"[INFO] Ingested {len(docs)} new conversation chunks (up to line {line_no}).")
            docs, metadatas, ids = [], [], []

    # Final flush
    if docs:
        embeddings = model.encode(docs, show_progress_bar=False).tolist()
        collection.add(
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"[INFO] Ingested {len(docs)} new conversation chunks (final batch).")

    # Update ingest state
    if new_last_line > last_line:
        save_state(new_last_line)
        print(f"[INFO] Updated ingest state: last_line={new_last_line}")
    else:
        print("[INFO] No new conversations to ingest.")


if __name__ == "__main__":
    main()

