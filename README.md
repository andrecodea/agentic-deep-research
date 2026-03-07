# Deep Research Agent

Enterprise-grade multi-agent research pipeline orchestrated with LangGraph. Features semantic memory via Chroma with Human-in-the-Loop (HITL) curation, session persistence via PostgreSQL, and full observability via LangSmith.

> Inference powered by Mercury 2 (Inception Labs) — a state-of-the-art diffusion LLM.

---

## Architecture

```
START
  ↓
Orchestrator — Classifies intent via LLM
  ↓
Router — Routes to retriever, researcher, or writer based on intent
  ↓
┌─────────────────────────────────────────────┐
│  Retriever — Chroma vector store query      │
│    ├── HIT (score < 0.3) → Writer           │
│    └── MISS → Researcher                    │
│                                             │
│  Researcher — Tavily search + extract       │
│    └── Writer                               │
└─────────────────────────────────────────────┘
  ↓
Writer — Generates Markdown report via LLM
  ↓
HITL — Human approval to save to Chroma
  ├── ACCEPT → Upserter → END
  └── IGNORE → END
```

### State

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `list[BaseMessage]` | Chat history (MessagesState) |
| `intent` | `Literal` | "conversation", "simple_search", "quick_search", "deep_research" |
| `research_mode` | `Literal` | "new", "existing", "quick" |
| `query` | `str` | Research query from user |
| `retrieved_docs` | `list[Document]` | Chroma results (add_docs reducer) |
| `cache_hit` | `bool` | Whether Chroma returned relevant context |
| `search_results` | `list[dict]` | Tavily results (add_docs reducer) |
| `final_report` | `str` | Generated report |
| `save_to_chroma` | `bool` | HITL decision flag |

### Project Structure

```
deep_research_agent/
├── src/
│   ├── agent.py              → StateGraph compilation
│   └── utils/
│       ├── state.py          → ResearchState (shared state + reducers)
│       ├── nodes.py          → orchestrator, router, retriever, researcher, writer, hitl, upserter
│       ├── tools.py          → Tavily search/extract, Chroma query/save
│       ├── prompts.py        → LangSmith Hub prompt retrieval
│       └── vectorstore.py    → Chroma singleton
├── tests/                    → pytest unit tests
├── langgraph.json            → LangGraph configuration
├── pyproject.toml            → Dependencies (uv)
└── .env                      → Environment variables
```

## Stack

- **Orchestration**: LangGraph (StateGraph)
- **LLM**: Mercury 2 (Inception Labs)
- **Search**: Tavily (search + extract)
- **Memory**: Chroma (vector store) + PostgreSQL (checkpointing)
- **Observability**: LangSmith

## Setup

```bash
# Install dependencies
uv sync

# Configure environment (.env)
INCEPTION_API_KEY=your_key
TAVILY_API_KEY=your_key
LANGSMITH_API_KEY=your_key
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Run locally
langgraph dev

# Or with Docker
docker compose up
```

Then open [LangGraph Studio](https://smith.langchain.com/studio) and point it to `http://127.0.0.1:2024`.

## Running Tests

```bash
pytest           # Run all tests
pytest -v        # Verbose
```

## Code Grade

**73/100** — Mid-level developer level

- Good state design and error handling
- Proper retry logic with tenacity
- Clean graph architecture
- Minor LSP warnings to address
- No test coverage yet
