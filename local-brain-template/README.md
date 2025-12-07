# Local Brain

Local Brain is a **self-hosted personal AI stack**:

- 🧠 RAG index (Chroma) over your own data  
- ⚙️ LM Studio as the local LLM engine  
- 🌐 FastAPI “Brain API” exposing an **OpenAI-compatible** interface  
- 💬 Open WebUI as the browser front-end  

All inference and data stay on your machine.

---

## Architecture

```text
                           ┌──────────────────────────┐
                           │        Open WebUI        │
                           │  Browser UI / Workspace  │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                         HTTP (OpenAI-compatible API)
                                        │
                           ┌────────────┴────────────┐
                           │      Brain API (v1)     │
                           │    FastAPI + uvicorn    │
                           └────────────┬────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
          ▼                             ▼                             ▼

   ┌───────────────┐           ┌────────────────┐           ┌──────────────────┐
   │ RAG Retrieval │◄──Query── │  Chroma Index  │           │ conversations.jsonl│
   │ (your code)   │           │ ./index        │           │ ./data            │
   └───────────────┘           └────────────────┘           └──────────────────┘

                                        │
                                        ▼
                              ┌────────────────┐
                              │ LM Studio API │
                              │ localhost:1234 │
                              └────────────────┘
                                        │
                                    Local Model
                             (e.g. gpt-oss-20b)


Prerequisites

You should be comfortable with:

	A terminal

	Python virtual environments

	Docker

	Basic networking concepts

You will need:
	
	Python 3.10+
	
	Git

	Docker (for Open WebUI)

	LM Studio (desktop app)


1. Install & Configure LM Studio

Download and install LM Studio from their website.

Open LM Studio.

Download a model compatible with the OpenAI-style server, e.g.:

gpt-oss-20b (or any other model you prefer)

Enable the OpenAI-compatible server in LM Studio:

Host: localhost

Port: 1234 (default; you can change it if you like)

Make sure it's running before proceeding.

You should be able to hit:

curl -s http://localhost:1234/v1/models

and get JSON back.

2. Clone Local Brain

git clone git@github.com:bealmot/local-brain.git
cd local-brain

3. Run the Interactive Setup Wizard

This script will:

Ask for LM Studio base URL & model name

Verify that LM Studio is reachable

Ask where to store:

chat exports (chat_export/)

index (index/)

data (data/)

Write config.yaml

Create the required directories

Optionally create requirements.txt

Run:

python3 setup_brain.py

Follow the prompts. Typical answers:

LM Studio base URL: http://localhost:1234/v1

Model ID: the ID you saw from /v1/models, e.g. openai/gpt-oss-20b

Base dir: the project directory (.)

Chat export dir: chat_export

Index dir: index

Data dir: data

API port: 8001

If LM Studio is not reachable at the URL you provide, the wizard will stop and tell you to start the server first.

4. Create & Activate Python Virtual Env

From the project root:

python3 -m venv rag
source rag/bin/activate


(Use .\rag\Scripts\activate on Windows PowerShell if needed.)

5. Install Python Dependencies

If you let the setup wizard create requirements.txt, then:

pip install -r requirements.txt

Otherwise:

pip install fastapi uvicorn chromadb pydantic pyyaml requests

6. Run the Brain API

From the project root with venv activated:

uvicorn brain_api:app --host 0.0.0.0 --port 8001


You should now be able to hit:

curl -s http://localhost:8001/health | jq
curl -s http://localhost:8001/v1/models | jq


and get JSON responses.

You can also test a completion:

curl -s -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<your-model-id>",
    "messages": [
      { "role": "user", "content": "Say hello from the Local Brain API." }
    ]
  }' | jq

7. Run Open WebUI (Docker)

If you don’t already have Open WebUI:

sudo docker run -d \
  --name openwebui \
  -p 3000:8080 \
  -v openwebui_data:/app/backend/data \
  ghcr.io/open-webui/open-webui:main


This will:

Pull the latest Open WebUI image

Start it on http://localhost:3000

Open your browser to:

http://localhost:3000

Create an admin user if prompted.

8. Connect Open WebUI to Local Brain

Inside Open WebUI:

Go to Settings → Admin → Connections (or Models / API settings depending on version).

Add a new OpenAI-compatible connection.

If Open WebUI is running in Docker with default bridge networking, your host is usually reachable as 172.17.0.1 from inside the container.

Use:

Base URL: http://172.17.0.1:8001/v1

API key: (can be anything, Local Brain ignores it)

Model IDs: leave blank to use /v1/models auto-discovery, or manually put the model ID you configured in config.yaml.

Save the connection.

In the main chat view:

Select this connection/model.

Ask something like:

“Test: reply with web-ok-1 and nothing else.”

If everything is wired correctly, the response will come from your Local Brain API using RAG + LM Studio.

9. Where Your Data Lives

By default (depending on your answers in the setup wizard):

chat_export/
Put your exported ChatGPT conversations or other text corpora here.
Your indexing scripts can read from this folder.

index/
Chroma / vector index files.

data/
Runtime data:

conversations.jsonl – logs of each interaction

future: profiles, summaries, metrics, etc.

These paths are stored in config.yaml, generated by setup_brain.py.

0. Next Steps / Roadmap

This template is focused on getting you to a working local brain quickly. Possible next extensions:

Automatic ingestion of:

new Open WebUI conversations

updated chat exports

local notes / projects

Daily profile synthesis:

Generate profile.md summarizing what the system knows about:

your environment

your projects

your preferences

Model routing:

Different models for:

coding

creative writing

system administration

uncensored exploration

Filesystem ingestion:

Index an Obsidian vault

Index config directories

Index code repositories

CLI tooling:

brainctl summarize last-week

brainctl list-projects

brainctl ingest chat-export
