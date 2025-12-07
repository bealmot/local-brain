#!/usr/bin/env python3
"""
setup_brain.py

Interactive setup wizard for Local Brain.

- Prints ASCII art banner
- Confirms LM Studio server is running
- Asks for LM Studio base URL + model
- Asks for basic paths (chat_export, index, data)
- Creates config.yaml
- Ensures directories exist
- Optionally writes requirements.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
import textwrap
import yaml
import requests


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


BANNER = r"""
  _                _        ____             _       
 | |    ___   __ _(_)_ __  | __ ) _ __ _ __(_)_ __  
 | |   / _ \ / _` | | '_ \ |  _ \| '__| '__| | '_ \ 
 | |__| (_) | (_| | | | | || |_) | |  | |  | | | | |
 |_____\___/ \__, |_|_| |_||____/|_|  |_|  |_|_| |_|
             |___/                                  

     Local Brain – Personal RAG + LLM stack
"""


def ask(prompt: str, default: str | None = None) -> str:
    if default is not None:
        full = f"{prompt} [{default}]: "
    else:
        full = f"{prompt}: "
    ans = input(full).strip()
    return ans or (default or "")


def yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{prompt} ({default_str}): ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def check_lm_studio(base_url: str) -> bool:
    """
    Quick sanity check that LM Studio's OpenAI-compatible server is up.

    We hit GET <base_url>/models (or /v1/models if user gave root).
    """
    url = base_url.rstrip("/")
    # If they gave http://host:port, append /v1/models
    if not url.endswith("/v1") and not url.endswith("/v1/"):
        models_url = url + "/models"
    else:
        models_url = url + "/models"

    try:
        resp = requests.get(models_url, timeout=5)
        if resp.status_code == 200:
            return True
        print(f"[WARN] LM Studio responded with status {resp.status_code} at {models_url}")
        return False
    except Exception as e:
        print(f"[ERROR] Could not reach LM Studio at {models_url}: {e}")
        return False


def main() -> int:
    print(BANNER)

    print("Welcome! This wizard will configure your Local Brain project.")
    print("Before you continue, make sure you have:")
    print("  1) Installed LM Studio")
    print("  2) Downloaded at least one model")
    print("  3) Started LM Studio's OpenAI-compatible server\n")

    if not yes_no("Have you already installed LM Studio and started its server?", True):
        print("Please install LM Studio, download a model, start the server, then re-run setup_brain.py.")
        return 1

    # --- LM Studio settings first (so we can verify) ---
    print("\nLM Studio API settings:")
    lm_base_url = ask("LM Studio base URL", "http://localhost:1234/v1")
    lm_model = ask("Default model ID", "openai/gpt-oss-20b")

    print("\nChecking LM Studio connectivity...")
    if not check_lm_studio(lm_base_url):
        print("\n❌ LM Studio does not appear to be reachable at that URL.")
        print("   Make sure the LM Studio server is running and the URL is correct, then try again.")
        return 1
    else:
        print("✅ LM Studio appears reachable.\n")

    # --- Base directory (project root) ---
    default_base = str(ROOT)
    base_dir = ask("Project base directory", default_base)
    base = Path(base_dir).expanduser().resolve()

    # --- Paths relative to base ---
    print("\nProject data paths (relative to base dir):")
    chat_export_dir = ask("Chat export directory (for ChatGPT exports)", "chat_export")
    index_dir = ask("Index directory (Chroma / vector index)", "index")
    data_dir = ask("Data directory (logs, JSONL, etc.)", "data")

    # --- API port ---
    api_port_str = ask("Brain API port", "8001")
    try:
        api_port = int(api_port_str)
    except ValueError:
        print("Invalid port; using 8001.")
        api_port = 8001

    # --- Confirm summary ---
    print("\nSummary:")
    print(f"  Base dir:        {base}")
    print(f"  chat_export_dir: {chat_export_dir}")
    print(f"  index_dir:       {index_dir}")
    print(f"  data_dir:        {data_dir}")
    print(f"  LM base_url:     {lm_base_url}")
    print(f"  LM model:        {lm_model}")
    print(f"  API port:        {api_port}")

    if not yes_no("\nDoes this look correct?", True):
        print("Aborting setup. No changes written.")
        return 1

    # --- Build config dict ---
    cfg = {
        "paths": {
            "base_dir": str(base),
            "chat_export_dir": chat_export_dir,
            "index_dir": index_dir,
            "data_dir": data_dir,
        },
        "lm_studio": {
            "base_url": lm_base_url.rstrip("/"),
            "model": lm_model,
        },
        "api": {
            "port": api_port,
        },
    }

    # --- Write config.yaml ---
    CONFIG_PATH.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\n✅ Wrote config.yaml → {CONFIG_PATH}")

    # --- Create directories ---
    for rel in (chat_export_dir, index_dir, data_dir):
        p = base / rel
        p.mkdir(parents=True, exist_ok=True)
        print(f"📁 Ensured directory exists: {p}")

    # --- requirements.txt (optional) ---
    if yes_no("\nCreate/update requirements.txt in this folder?", True):
        req_path = ROOT / "requirements.txt"
        base_reqs = textwrap.dedent(
            """\
            fastapi
            uvicorn
            chromadb
            pydantic
            pyyaml
            requests
            """
        )
        if req_path.exists():
            print(f"requirements.txt already exists at {req_path}; leaving as-is.")
        else:
            req_path.write_text(base_reqs, encoding="utf-8")
            print(f"✅ Wrote basic requirements.txt → {req_path}")

    print("\n🎉 Setup complete!")
    print("Next steps (typical):")
    print(f"  1) Create/activate a venv:")
    print(f"       python -m venv rag && source rag/bin/activate")
    print(f"  2) Install dependencies:")
    print(f"       pip install -r requirements.txt")
    print(f"  3) Run the Brain API:")
    print(f"       uvicorn brain_api:app --host 0.0.0.0 --port {api_port}")
    print("  4) Run Open WebUI via Docker and point it at:")
    print(f"       http://<host-ip>:{api_port}/v1")
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
