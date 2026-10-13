# Agentic Research Analyst

A multi-agent research system that takes a research question, searches the web, extracts structured insights, critiques its own findings, and generates a professional report — all running on local LLMs via Ollama with no external API dependencies in the core pipeline.

---

## What it does

1. **Planner** — breaks your query into 4 specific, searchable tasks
2. **Researcher** — extracts concrete facts from pre-fetched web content using RAG
3. **Critic** — identifies gaps, unsupported claims, and missing angles
4. **Reporter** — synthesises everything into a structured 6-section report
5. **Follow-up support** — subsequent queries build on the previous report's context, restored from disk on startup
6. **Google Drive export** — saves report to Drive via MCP (requires Anthropic API key)

---

## Project structure

```
agentic-research-analyst/
│
├── app.py                        # Streamlit UI and pipeline orchestration
├── memory.py                     # Persistent JSON-based memory store
├── pyproject.toml                # Dependencies (managed with UV)
│
├── crew/
│   ├── agents.py                 # CrewAI agent definitions
│   ├── tasks.py                  # Task descriptions and format rules
│   └── research_crew.py          # Crew orchestration and web fetch
│
├── agents/
│   ├── clarification_agent.py    # Blocks vague queries before crew runs
│   └── search_agent.py           # DuckDuckGo search and result ranking
│
├── utils/
│   ├── ollama_client.py          # Shared Ollama wrapper
│   ├── web_scraper.py            # BeautifulSoup page scraper
│   ├── json_parser.py            # Safe JSON parsing with retry
│   └── drive_saver.py            # Google Drive export via Anthropic MCP
│
├── rag/
│   └── vector_store.py           # ChromaDB with local nomic-embed-text embeddings
│
└── examples/
    └── sample_report.txt         # Example output from one run
```

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running locally
- The following models pulled in Ollama:

```
ollama pull llama3.1        # researcher and critic (better tool-use than llama3)
ollama pull mistral         # planner and reporter (reliable structured output)
ollama pull nomic-embed-text  # local embeddings for RAG
```

---

## Setup

```
# 1. Clone the repo
git clone https://github.com/smerarawal/Agentic-Research-Analyst-by-Ollama
cd Agentic-Research-Analyst-by-Ollama

# 2. Install UV if you don't have it
pip install uv

# 3. Create venv and install dependencies
uv sync

# 4. Set Ollama's default context window, then start the server.
#    Ollama's documented server variable is OLLAMA_CONTEXT_LENGTH.
#    If the Ollama desktop app is already running, quit it first, or set the
#    variable permanently in your system environment variables and restart Ollama.
#    cmd:         set OLLAMA_CONTEXT_LENGTH=16384
#    PowerShell:  $env:OLLAMA_CONTEXT_LENGTH=16384
ollama serve

# 5. Run the app in a second terminal
streamlit run app.py
```

---

## Usage

1. Open the Streamlit app in your browser (default: `http://localhost:8501`)
2. Enter a specific research query and click **Run Research**
3. View per-agent outputs in the expandable sections
4. Download the report or save to Google Drive
5. For follow-ups, enter a new query — previous context is shown at the top and carried forward automatically

### Query tips

| Too vague — will be blocked | Specific enough — will run                                       |
| --------------------------- | ---------------------------------------------------------------- |
| `analyze NVIDIA`            | `NVIDIA Blackwell GPU data center deployment 2024`               |
| `research Apple`            | `Apple Vision Pro enterprise adoption and competitor comparison` |
| `study Tesla`               | `Tesla FSD regulatory approval timeline and safety record`       |

---

## Models and roles

| Agent      | Model              | Why                                            |
| ---------- | ------------------ | ---------------------------------------------- |
| Planner    | `mistral`          | Reliable structured output and JSON formatting |
| Researcher | `llama3.1`         | Better tool-use support than base llama3       |
| Critic     | `llama3.1`         | Stronger analytical reasoning                  |
| Reporter   | `mistral`          | Consistent structured report formatting        |
| Embeddings | `nomic-embed-text` | Local semantic embeddings, no API key needed   |

---

## Key design decisions

**Why pre-fetch content instead of tool calling?** Local models (llama3, mistral) misformat tool call arguments when used via CrewAI, causing repeated failures. Web content is fetched and scraped before the crew starts, then passed directly into the task. llama3.1 has improved tool-use support and may restore autonomous tool calling — swap it in and test if needed.

**Why local embeddings for RAG?** CrewAI's built-in memory (`memory=True`) defaults to OpenAI embeddings. Using `nomic-embed-text` via Ollama achieves the same semantic retrieval with no external API dependency. Scraped content is chunked at 300 words with 30-word overlap and embedded into ChromaDB on each run.

**Why persistent JSON memory?** The original in-memory list reset on every browser refresh. The JSON store writes each report to disk with a timestamp and query label, so follow-up context survives restarts. A **Clear Memory** button in the UI wipes the file for fresh sessions.

**Why raise the context window to 16384?** The default Ollama context window is 4096 tokens. Research prompts with scraped content regularly exceed this, causing 600s timeouts. Setting `OLLAMA_CONTEXT_LENGTH=16384` gives the model 4× the room without needing to cut content as aggressively.

**Why mistral for the reporter?** Mistral follows structured format instructions more consistently than llama3 for the final report. llama3 tends to leak chain-of-thought prefixes (`"I now can give a great answer"`) as its output — mistral produces clean structured text.

---

## Limitations

- Tool calling is unreliable with base llama3/mistral — llama3.1 improves this but GPT-4 handles it most reliably
- Paywalled sites (Reuters, Bloomberg, WSJ) return empty content — pipeline falls back to DuckDuckGo snippets
- Google Drive export requires an Anthropic API key with active credits
- ChromaDB runs in-memory within each session — embeddings do not persist across server restarts
- Local inference on CPU takes 2–3 minutes per full run

---

## Built with

- [Streamlit](https://streamlit.io) — UI
- [CrewAI](https://www.crewai.com) — multi-agent orchestration
- [Ollama](https://ollama.com) — local LLM inference
- [ChromaDB](https://www.trychroma.com) — vector store for RAG
- [ddgs](https://pypi.org/project/ddgs/) — DuckDuckGo search
- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/) — web scraping
- [UV](https://github.com/astral-sh/uv) — package management
