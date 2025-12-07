#!/usr/bin/env python3
import pathlib
import yaml
import requests

HERE = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    lm = cfg["lm_studio"]
    base_url = lm["base_url"].rstrip("/")
    api_key = lm["api_key"]
    model = lm["model"]

    url = f"{base_url}/chat/completions"
    print(f"[INFO] Pinging LM Studio at {url} with model '{model}'")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a diagnostic helper."},
            {"role": "user", "content": "Reply with: LM STUDIO OK"},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        print("[INFO] Response:", answer)
    except Exception as e:
        print("[ERROR] Failed to reach LM Studio:", repr(e))


if __name__ == "__main__":
    main()
