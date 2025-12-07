#!/usr/bin/env python3
import os
import json
import pathlib
from typing import List, Dict, Any

import yaml
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


HERE = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def iter_chatgpt_messages(export_dir: pathlib.Path):
    """
    Very simple loader that tries to handle the common OpenAI ChatGPT export format:
    - conversations.json
    - or multiple JSON files in a 'conversations' subdir

    It yields (conversation_title, message_role, message_text).
    """
    conversations_file = export_dir / "conversations.json"
    files = []

    if conversations_file.exists():
        files.append(conversations_file)
    else:
        # Fallback: any *.json under export_dir
        files.extend(export_dir.glob("**/*.json"))

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue

        # Two common shapes:
        # 1) {"title": ..., "mapping": {...}} style (older exports)
        # 2) list of conversation dicts
        if isinstance(data, dict) and "mapping" in data:
            title = data.get("title", path.stem)
            mapping = data["mapping"].values()
            for node in mapping:
                msg = node.get("message")
                if not msg:
                    continue
                role = msg.get("author", {}).get("role", "unknown")
                content_parts = msg.get("content", {}).get("parts") or []
                text = "\n\n".join(
                    p for p in content_parts if isinstance(p, str)
                ).strip()
                if text:
                    yield title, role, text

        elif isinstance(data, list):
            # List of conversations
            for conv in data:
                title = conv.get("title", path.stem)
                mapping = conv.get("mapping", {}).values()
                for node in mapping:
                    msg = node.get("message")
                    if not msg:
                        continue
                    role = msg.get("author", {}).get("role", "unknown")
                    content_parts = msg.get("content", {}).get("parts") or []
                    text = "\n\n".join(
                        p for p in content_parts if isinstance(p, str)
                    ).strip()
                    if text:
                        yield title, role, text
        else:
            print(f"[WARN] Unrecognized JSON shape in {path}")


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Naive character-based chunking with sentence-ish boundaries."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks = []
    current = []

    for line in text.split("\n"):
        if sum(len(x) for x in current) + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current).strip())
            current = []
        current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    return chunks


def build_index():
    cfg = load_config()
    export_dir = pathlib.Path(cfg["chatgpt_export_dir"]).expanduser()
    index_dir = pathlib.Path(cfg["index_dir"]).expanduser()
    collection_name = cfg["rag"]["collection_name"]

    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using export dir: {export_dir}")
    print(f"[INFO] Using index dir:  {index_dir}")

    client = chromadb.PersistentClient(
        path=str(index_dir),
        settings=Settings(allow_reset=False)
    )

    collection = client.get_or_create_collection(name=collection_name)

    print(f"[INFO] Clearing existing chatgpt_export docs in '{collection_name}'")
    try:
        # Only delete docs that came from the ChatGPT export,
        # keep anything else (like live_cli ingested later)
        collection.delete(where={"source": "chatgpt_export"})
    except Exception:
        pass

    # Embedding model
    print("[INFO] Loading embedding model (this can take a moment)…")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    docs: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    doc_id = 0
    for title, role, text in iter_chatgpt_messages(export_dir):
        for chunk in chunk_text(text):
            docs.append(chunk)
            metadatas.append(
                {
                    "source": "chatgpt_export",
                    "conversation_title": title,
                    "role": role,
                }
            )
            ids.append(f"chatgpt-{doc_id}")
            doc_id += 1

            if len(docs) >= 128:
                # Batch insert
                embeddings = model.encode(docs, show_progress_bar=False).tolist()
                collection.add(
                    documents=docs,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                print(f"[INFO] Indexed {doc_id} chunks…")
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
        print(f"[INFO] Indexed {doc_id} chunks total.")

    print("[INFO] Index build complete.")


if __name__ == "__main__":
    build_index()
