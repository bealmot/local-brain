Local Brain

Local Brain is a self-hosted personal AI stack designed to give you a private, persistent, retrieval-augmented assistant that runs entirely on your own machine.

It integrates:
- RAG index (ChromaDB) built over your personal data
- LM Studio as the local LLM inference engine
- Brain API (FastAPI) exposing an OpenAI-compatible /v1/ interface
- Open WebUI as the front-end workspace

Everything runs locally. Nothing leaves your machine.

-----------------------------------------------------------------------
ARCHITECTURE

Open WebUI (browser UI)
        │
        ▼
Brain API (FastAPI + uvicorn, exposes OpenAI-compatible endpoints)
        │
        ├── RAG retrieval (your Python code)
        │
        ├── Chroma vector index (./index)
        │
        └── Logged conversations (./data/conversations.jsonl)
        │
        ▼
LM Studio API (localhost:1234)
Local Model (e.g. gpt-oss-20b)

-----------------------------------------------------------------------
PREREQUISITES

You should be comfortable with:
- Terminal use
- Python virtual environments
- Docker
- Basic networking

You need:
- Python 3.10+
- Git
- Docker
- LM Studio desktop app

-----------------------------------------------------------------------
1. INSTALL AND CONFIGURE LM STUDIO

1. Install LM Studio.
2. Launch LM Studio.
3. Download a model (example: gpt-oss-20b).
4. Start the OpenAI-Compatible Server:
     Host: localhost
     Port: 1234
5. Verify that it is running:
     curl -s http://localhost:1234/v1/models

If you see JSON, LM Studio is working.

-----------------------------------------------------------------------
2. CLONE LOCAL BRAIN

git clone git@github.com:bealmot/local-brain.git
cd local-brain

-----------------------------------------------------------------------
3. RUN THE INTERACTIVE SETUP WIZARD

The setup wizard does the following:
- Asks for LM Studio base URL
- Verifies LM Studio is reachable
- Asks for:
     chat_export directory
     index directory
     data directory
- Creates the directories automatically
- Writes config.yaml
- Optionally writes requirements.txt

Run:
python3 setup_brain.py

Typical values:
LM Studio base URL: http://localhost:1234/v1
Model ID: something like openai/gpt-oss-20b
Base dir: .
Chat export dir: chat_export
Index dir: index
Data dir: data
API port: 8001

The wizard will stop if LM Studio cannot be reached.

-----------------------------------------------------------------------
4. CREATE AND ACTIVATE PYTHON VIRTUAL ENVIRONMENT

python3 -m venv rag
source rag/bin/activate

Windows PowerShell:
.\rag\Scripts\activate

-----------------------------------------------------------------------
5. INSTALL DEPENDENCIES

If requirements.txt exists:
pip install -r requirements.txt

Otherwise:
pip install fastapi uvicorn chromadb pydantic pyyaml requests

-----------------------------------------------------------------------
6. RUN THE BRAIN API

uvicorn brain_api:app --host 0.0.0.0 --port 8001

Tests:
curl -s http://localhost:8001/health
curl -s http://localhost:8001/v1/models

Test a completion:
curl -s -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "<your-model-id>",
        "messages": [
          {"role": "user", "content": "Say hello from the Local Brain API."}
        ]
      }'

-----------------------------------------------------------------------
7. RUN OPEN WEBUI (DOCKER)

sudo docker run -d \
  --name openwebui \
  -p 3000:8080 \
  -v openwebui_data:/app/backend/data \
  ghcr.io/open-webui/open-webui:main

Open your browser:
http://localhost:3000

Create an admin account if needed.

-----------------------------------------------------------------------
8. CONNECT OPEN WEBUI TO THE LOCAL BRAIN API

Inside Open WebUI:
Settings → Admin → Connections

Add a new OpenAI-compatible connection.

If WebUI is running through Docker bridge networking, your host is likely reachable at:
172.17.0.1

Use:
Base URL: http://172.17.0.1:8001/v1
API key: anything (ignored)
Model IDs: leave blank for auto-discovery or specify manually

Save.

Test:
Ask: "Reply with web-ok-1"

If it responds correctly, the Local Brain pipeline is working.

-----------------------------------------------------------------------
9. WHERE YOUR DATA LIVES

chat_export/
- Put your exported ChatGPT logs or other text files here.

index/
- Chroma vector index files.

data/
- Conversations logged as JSONL.
- Future summaries, profiles, metadata.

These paths come from config.yaml created by setup_brain.py.

-----------------------------------------------------------------------
10. ROADMAP / NEXT STEPS

Automatic ingestion:
- New WebUI conversations
- Updated chat exports
- Local notes and documents

Daily profile synthesis:
- Generate profile.md summarizing your environment, habits, and projects

Model routing:
- Use different models for coding, sysadmin, creative tasks, etc.

Filesystem ingestion:
- Index Obsidian vaults
- Index config directories
- Index source code

CLI tooling:
- brainctl summarize last-week
- brainctl list-projects
- brainctl ingest chat-export

-----------------------------------------------------------------------
END OF README
