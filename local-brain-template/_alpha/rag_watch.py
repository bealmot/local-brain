#!/usr/bin/env python3
import time
import pathlib
from typing import Any, Dict

import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# For simplicity we just call the full build_index() for now
from rag_index import build_index, CONFIG_PATH


class ExportChangeHandler(FileSystemEventHandler):
    def __init__(self, export_dir: pathlib.Path):
        super().__init__()
        self.export_dir = export_dir

    def on_any_event(self, event):
        # Debounce could be added, but for now we just rebuild fully.
        print(f"[EVENT] Change detected: {event.src_path}")
        print("[INFO] Rebuilding index due to change…")
        build_index()


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    export_dir = pathlib.Path(cfg["chatgpt_export_dir"]).expanduser()

    event_handler = ExportChangeHandler(export_dir=export_dir)
    observer = Observer()
    observer.schedule(event_handler, str(export_dir), recursive=True)
    observer.start()

    print(f"[INFO] Watching {export_dir} for changes. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Stopping watcher…")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
