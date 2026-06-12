# Lightweight RAG Chatbot

A document-grounded AI assistant that ingests business documents (PDF, DOCX, CSV, Excel, TXT/MD), retrieves relevant context with hybrid search, and answers with citations. Built for the SutraAI assignment: lightweight, reliable, and safe to fail when evidence is weak.

## Features

- **Multi-backend LLM** — Ollama (local), OpenAI, Gemini, Hugging Face, or NVIDIA NIM (sidebar switcher)
- **Hybrid RAG** — dense embeddings + BM25 keyword search, merged with RRF, reranked with a cross-encoder
- **Document ingestion** — parsers for PDF, DOCX, CSV, Excel, plain text; LLM metadata enrichment (summary, keywords, intent tags)
- **Corpus-driven routing** — skips full retrieval for general/chitchat queries; uses sidecar keywords/tags for document-shaped questions
- **Guardrails** — low-confidence abstention, prompt-injection sanitization, citation enforcement
- **Clarification flow** — asks one disambiguation question when retrieval finds multiple equally relevant sections
- **Chat persistence** — SQLite-backed conversation history with sidebar to browse, switch, and delete chats
- **Multi-turn memory** — prior turns passed to the LLM for follow-up questions

## Architecture (high level)

```
User query
  → RAG router (corpus keywords / semantic match)
  → [optional] retrieve_hybrid (Chroma + BM25 + RRF + reranker)
  → guardrails (abstain / clarify / sanitize)
  → LLM (grounded system prompt + chat history)
  → answer + source citations
```

**Key paths**


| Path                         | Purpose                                      |
| ---------------------------- | -------------------------------------------- |
| `main.py`                    | Streamlit UI                                 |
| `assistant/rag.py`           | Ingestion, ChromaDB, hybrid retrieval        |
| `assistant/chatbot.py`       | Routing, guardrails, response orchestration  |
| `assistant/chat_store.py`    | SQLite conversations & messages              |
| `assistant/router.py`        | Decide if query needs document retrieval     |
| `assistant/guardrails.py`    | Abstention, injection screen, citation check |
| `assistant/clarification.py` | Ambiguous-query disambiguation               |
| `data/`                      | Source documents                             |
| `chroma_db/`                 | Vector index (generated, gitignored)         |
| `data/chats.db`              | Chat history (generated, gitignored)         |


## Setup

```bash
conda create --prefix venv python=3.11
conda activate ./venv
pip install -r requirements.txt
```

Copy environment variables into `.env` (see below). **Do not commit `.env`.**

### Ingest sample documents

Place files in `data/`, then:

```bash
python ingest_document.py
```

Or upload via the Streamlit sidebar (**Documents** → **Ingest**).

### Run the app

```bash
streamlit run main.py
```

Use the sidebar to pick a backend/model, manage conversations, and upload new files.

## LLM backends


| Backend        | Env variable                                      | Notes                                                            |
| -------------- | ------------------------------------------------- | ---------------------------------------------------------------- |
| Ollama (local) | optional `OLLAMA_MODEL`                           | `ollama pull tinyllama`                                          |
| OpenAI         | `OPENAI_API_KEY`, optional `OPENAI_MODEL`         | default `gpt-4o-mini`                                            |
| Gemini         | `GEMINI_API_KEY`                                  | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Hugging Face   | `HF_TOKEN`                                        | Inference Providers enabled                                      |
| NVIDIA NIM     | `NVIDIA_NIM_API_KEY`, optional `NVIDIA_NIM_MODEL` | [build.nvidia.com](https://build.nvidia.com)                     |


### Programmatic usage

```python
from llm.infer import LLM

llm = LLM("openai")
print(llm.chat("Hello", system="You are a helpful assistant."))

history = [
    {"role": "user", "content": "What is the remote work policy?"},
    {"role": "assistant", "content": "Employees may work remotely per HR-RWP-2026-004."},
]
for chunk in llm.stream_messages(history + [{"role": "user", "content": "What about stipends?"}]):
    print(chunk, end="")
```

## Guardrails


| Guardrail            | Why                                              | Limitation                            |
| -------------------- | ------------------------------------------------ | ------------------------------------- |
| **RAG router**       | Avoids irrelevant retrieval on chitchat          | Threshold tuning per corpus           |
| **Abstention**       | Refuses when reranker confidence is too low      | `MIN_USEFUL_SCORE` in `guardrails.py` |
| **Injection screen** | Strips adversarial phrases from retrieved chunks | Regex-based                           |
| **Citation check**   | Warns when RAG answers lack source tags          | Structural, not semantic              |


## Assumptions

- English documents, single-user local deployment
- Corpus fits in memory (assignment scale, not production)
- At least one cloud or local LLM API key for enrichment and chat

## Limitations

- CSV/Excel rows are embedded as text
- DOCX parsing is paragraph-level (no heading hierarchy yet)
- Cross-encoder + embedding models download on first run (~100 MB)
- Evaluation harness not yet implemented

## Future improvements

- Structured-data query path (`pandas` over ingested tables)
- Lightweight `eval/` script for retrieval and groundedness checks

## AI assistance disclosure

AI coding tools (LLM assistants) were used during development for:

- **README creation** — structure, setup docs, and assignment-aligned sections
- **UI creation** — Streamlit layout, conversation sidebar, sources panel
- **Brainstorming** — planning the best approach to create light weight rag chatbot
- **Debug** - review code for any errors

All implementation was reviewed and integrated into this repository; API keys and local data are not included in version control.