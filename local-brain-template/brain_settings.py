#!/usr/bin/env python3
"""
brain_settings.py

Single source of truth for paths and config values.

All other scripts should import from here instead of hardcoding paths.
"""

from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


class BrainConfig:
    def __init__(self, data: dict):
        self._data = data

    @property
    def raw(self) -> dict:
        return self._data

    # --- Paths ---
    @property
    def base_dir(self) -> Path:
        base = self._data.get("paths", {}).get("base_dir", ".")
        return Path(base).expanduser().resolve()

    @property
    def chat_export_dir(self) -> Path:
        p = self._data.get("paths", {}).get("chat_export_dir", "chat_export")
        return self.base_dir / p

    @property
    def index_dir(self) -> Path:
        p = self._data.get("paths", {}).get("index_dir", "index")
        return self.base_dir / p

    @property
    def data_dir(self) -> Path:
        p = self._data.get("paths", {}).get("data_dir", "data")
        return self.base_dir / p

    # --- LM Studio ---
    @property
    def lm_studio_base_url(self) -> str:
        return self._data.get("lm_studio", {}).get("base_url", "http://localhost:1234/v1")

    @property
    def lm_studio_model(self) -> str:
        return self._data.get("lm_studio", {}).get("model", "openai/gpt-oss-20b")

    # --- OpenWebUI / API ---
    @property
    def api_port(self) -> int:
        return int(self._data.get("api", {}).get("port", 8001))


def load_brain_config() -> BrainConfig:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return BrainConfig(data)
