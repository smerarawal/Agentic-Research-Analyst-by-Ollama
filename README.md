# Gemini Research Analyst

A Gemini API research prototype that plans web searches, extracts claims with source quotes, asks a critic to find gaps, and writes a cited report. Search and page retrieval use the web; prompts and retrieved source text are sent to Google's Gemini API. Automated checks reduce some errors but do not prove that a report is true.

---

## What it does

1. **Clarify and plan** — vague queries can end with a question; specific queries become searchable sub-questions.
2. **Search and retrieve** — search results are cached, pages are checked against `robots.txt`, and text is indexed in a per-run Chroma collection.
3. **Extract evidence** — Gemini returns claims with source IDs and exact quotes; code drops quotes absent from retrieved text.
4. **Critique loop** — the critic can request up to two additional search rounds.
5. **Write and check** — code checks source IDs, headings, quote-backed claims, and numbers. A failed first check triggers one rewrite.
6. **Trace and memory** — per-run JSON traces include node times and Gemini token counts. Only verified facts are stored for later retrieval.

---

## Project structure

```
agentic-research-analyst/
│
├── app.py                        # Streamlit UI and pipeline orchestration
├── workflow.py                   # LangGraph nodes, routing, retrieval, checks, traces
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
│   ├── ollama_client.py          # Legacy Ollama wrapper (not used by Day 3 graph)
│   ├── web_scraper.py            # BeautifulSoup page scraper
│   ├── json_parser.py            # Safe JSON parsing with retry
│   └── drive_saver.py            # Google Drive export via Anthropic MCP
│
├── rag/
│   └── vector_store.py           # Legacy vector-store code
├── tests/
│   └── test_workflow.py          # Evidence checks and graph-routing tests
│
└── examples/
    └── sample_report.txt         # Example output from one run
```

---

## Requirements

- Python 3.12+
- A Gemini API key in `GEMINI_API_KEY`
- Internet access for search and Gemini API calls
- Gemini models default to `gemini-2.5-flash` and `gemini-embedding-001`; override with `GEMINI_MODEL` and `GEMINI_EMBED_MODEL`

Gemini calls send the query, retrieved web-source text, and embedding inputs to Google's API. Do not enter sensitive or confidential material.

---

## Setup

```
# 1. Clone the repo
git clone https://github.com/smerarawal/Agentic-Research-Analyst-by-Ollama
cd Agentic-Research-Analyst-by-Ollama

# 2. Install UV if you don't have it
pip install uv

# 3. Create venv and install dependencies
uv sync --extra dev

# 4. Set your Gemini API key in PowerShell for this terminal session.
$env:GEMINI_API_KEY = "your-key-here"

# 5. Start the app in this same terminal so it inherits the API key
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
| Graph nodes | `gemini-2.5-flash` | Structured planning, fact extraction, critique, and report generation through Gemini API |
| Embeddings | `gemini-embedding-001` | Semantic retrieval through Gemini API |

---

## Key design decisions

**Why a graph instead of a fixed sequence?** The critic can send the graph back to search when it finds evidence gaps. Search rounds and report rewrites have hard limits so the workflow terminates.

**Why embeddings for RAG?** `gemini-embedding-001` maps source chunks and questions to vectors used for semantic retrieval. Pages are split into 280-word chunks with 30-word overlap. Each run has a separate persistent Chroma collection so page chunks do not leak across queries. Source chunks and queries are sent to the Gemini API for embeddings.

**What does quote verification establish?** It establishes that an extracted quote appears in a retrieved chunk after whitespace normalization. It does not prove that the source is correct or that the quote logically supports the model's claim.

Gemini handles inference remotely, so local CPU inference speed is no longer the bottleneck. API availability, quota, network latency, and service limits still apply. Embedding requests are split into batches of 8.

---

## Limitations

- Automated quote and citation checks reduce some errors but do not prove factual correctness; review claims against the linked pages.
- Search snippets can be used when a page cannot be retrieved and are weaker evidence than the full page.
- The number check is string-based and can flag formatting differences.
- The integrated live search, page extraction, Chroma, and Gemini flow must be verified with a configured API key.
- Reports can contain errors even when automated citation and quote checks pass.

---

## Built with

- [Streamlit](https://streamlit.io) — UI
- [LangGraph](https://langchain-ai.github.io/langgraph/) — graph orchestration with bounded routing
- [Google Gemini API](https://ai.google.dev/gemini-api/docs) — text generation and embeddings
- [ChromaDB](https://www.trychroma.com) — vector store for RAG
- [ddgs](https://pypi.org/project/ddgs/) — DuckDuckGo search
- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/) — web scraping
- [Trafilatura](https://pypi.org/project/trafilatura/) — article text extraction
- [UV](https://github.com/astral-sh/uv) — package management
