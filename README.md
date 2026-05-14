# Agentic Research Analyst

A multi-agent research system that takes a research question, searches the web, extracts structured insights, critiques its own findings, and generates a professional report — all running on local LLMs via Ollama.

---

## What it does

1. **Planner** — breaks your query into specific, searchable tasks
2. **Search** — queries DuckDuckGo and scrapes full page content from open sources
3. **Extractor** — pulls concrete facts from scraped content using RAG (ChromaDB)
4. **Critic** — identifies gaps, unsupported claims, and missing angles
5. **Report Generator** — synthesises everything into a structured 6-section report
6. **Follow-up support** — subsequent queries build on the previous report's context

---

## Project structure

```
agentic-research-analyst/
│
├── app.py                  # Streamlit UI and pipeline orchestration
├── memory.py               # In-session memory store
├── workflow.py             # LangGraph workflow definition
├── pyproject.toml          # Dependencies (managed with UV)
│
├── agents/
│   ├── planner.py          # Breaks query into search tasks
│   ├── search_agent.py     # DuckDuckGo search + domain scoring
│   ├── extractor.py        # Web scraping + fact extraction
│   ├── critic.py           # Critiques extracted findings
│   ├── report_generator.py # Final report synthesis
│   └── clarification_agent.py  # Blocks vague queries
│
├── utils/
│   ├── ollama_client.py    # Shared Ollama wrapper
│   ├── web_scraper.py      # BeautifulSoup page scraper
│   └── json_parser.py      # Safe JSON parsing with retry
│
└── rag/
    └── vector_store.py     # ChromaDB add and retrieve
```

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running locally
- The following models pulled in Ollama:

```bash
ollama pull llama3
ollama pull mistral
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/agentic-research-analyst
cd agentic-research-analyst

# 2. Install UV if you don't have it
pip install uv

# 3. Create venv and install dependencies
uv sync

# 4. Run the app
streamlit run app.py
```

---

## Usage

1. Open the Streamlit app in your browser (default: `http://localhost:8501`)
2. Enter a specific research query
3. Click **Run Research**
4. For follow-ups, enter a new query — the previous report is carried forward as context

### Query tips

| Too vague — will be blocked | Specific enough — will run |
|---|---|
| `analyze NVIDIA` | `NVIDIA AI chip strategy in data center market 2024` |
| `research Apple` | `Apple Vision Pro enterprise adoption and competitor comparison` |
| `study Tesla` | `Tesla FSD regulatory approval timeline and safety record` |

---

## Models used

| Agent | Model | Why |
|---|---|---|
| Planner, Clarification | `mistral` | Reliable structured JSON output |
| Extractor, Critic, Report | `llama3` | Stronger long-form reasoning |

---

## Key design decisions

**Why local LLMs?**
No API costs, no data leaving your machine, and a good forcing function for writing prompts that work reliably without hosted-model quality crutches.

**Why parallel scraping?**
Sequential scraping at 5 pages × timeout = significant wall time. `ThreadPoolExecutor` runs all page fetches simultaneously, cutting scrape time from ~20s to ~4s.

**Why ChromaDB for RAG?**
Extracted text is embedded and stored each run, then retrieved semantically when generating the report. This means the report generator gets the most relevant chunks rather than blindly using everything — important when scraped content is noisy.

**Why mistral for the planner?**
Mistral follows JSON format instructions more consistently than llama3 on structured output tasks. The planner and clarification agent both need reliable JSON — everything else benefits from llama3's stronger reasoning.

---

## Limitations

- Memory resets on browser refresh (in-session only)
- Paywalled sites (Reuters, Bloomberg, WSJ) return empty content — the pipeline falls back to DuckDuckGo snippets for those
- Local model output quality is lower than GPT-4/Claude for complex synthesis — prompt specificity matters a lot
- ChromaDB runs in-memory, so RAG context does not persist across runs

---

## Built with

- [Streamlit](https://streamlit.io) — UI
- [Ollama](https://ollama.com) — local LLM inference
- [ChromaDB](https://www.trychroma.com) — vector store for RAG
- [ddgs](https://pypi.org/project/ddgs/) — DuckDuckGo search
- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/) — web scraping
- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent workflow orchestration
- [UV](https://github.com/astral-sh/uv) — package management