# Deep Research Agent

Multi-step graph research pipeline with Chainlit chat UI, orchestrated with LangGraph. Features semantic memory via Chroma with Human-in-the-Loop (HITL) curation, session persistence via PostgreSQL, and full observability via LangSmith.

> Inference powered by Mercury 2 (Inception Labs) — a state-of-the-art diffusion LLM.

---

## Architecture

```
User Message (Chainlit)
  ↓
START → Orchestrator — Classifies intent via LLM
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
Writer — Generates Markdown report via LLM (with HITL middleware)
  ↓
END → Response streamed to Chainlit UI
```

### State

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `list[BaseMessage]` | Chat history (MessagesState) |
| `intent` | `Literal` | "conversation", "simple_search", "quick_search", "deep_research" |
| `research_mode` | `Literal` | "new", "existing", "quick" |
| `query` | `str` | Research query from user |
| `topic` | `str` | Search topic context ("general", "news", "finance") |
| `retrieved_docs` | `list[Document]` | Chroma results (add_docs reducer) |
| `cache_hit` | `bool` | Whether Chroma returned relevant context (< 0.3 score) |
| `search_results` | `list[dict]` | Tavily results (add_docs reducer) |
| `final_report` | `str` | Generated report |
| `save_to_chroma` | `bool` | HITL decision flag |

### Project Structure

```
deep_research_agent/
├── src/
│   ├── main.py              → Chainlit entry point (chat UI)
│   ├── graph.py             → StateGraph compilation
│   └── utils/
│       ├── state.py         → ResearchState (shared state + reducers)
│       ├── nodes.py         → orchestrator, retriever, researcher, writer
│       ├── tools.py         → Tavily search/extract, Chroma query/save
│       ├── prompts.py       → LangSmith Hub prompt retrieval
│       └── vectorstore.py   → Chroma singleton
├── tests/                   → pytest unit tests
├── chainlit.md              → Chainlit welcome screen
├── langgraph.json           → LangGraph CLI configuration
├── pyproject.toml           → Dependencies (uv)
└── .env                     → Environment variables
```

## Stack

- **UI**: Chainlit (chat interface with streaming + auth)
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
INCEPTION_API_KEY=your_key          # Mercury 2 LLM
TAVILY_API_KEY=your_key             # Web search
LANGSMITH_API_KEY=your_key          # Observability
LANGGRAPH_DATABASE_URL=postgresql://user:pass@localhost:5432/db  # Session persistence
ADMIN_USER=your_admin_user          # Chainlit auth
ADMIN_PASS=your_admin_pass          # Chainlit auth
CHAINLIT_AUTH_SECRET=your_secret    # Chainlit session encryption

# Run with Chainlit (recommended)
chainlit run src/main.py

# Or run with LangGraph CLI
langgraph dev
```

Then open:
- **Chainlit**: http://localhost:8000 (with auth)
- **LangGraph Studio**: https://smith.langchain.com/studio → point to `http://127.0.0.1:2024`

## Running Tests

```bash
pytest           # Run all tests
pytest -v        # Verbose
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `INCEPTION_API_KEY` | Yes | Mercury 2 API key from Inception Labs |
| `TAVILY_API_KEY` | Yes | Tavily API key for web search |
| `LANGSMITH_API_KEY` | No | LangSmith for observability/tracing |
| `LANGGRAPH_DATABASE_URL` | Yes | PostgreSQL connection for session persistence |
| `ADMIN_USER` | Yes | Chainlit admin username |
| `ADMIN_PASS` | Yes | Chainlit admin password |
| `CHAINLIT_AUTH_SECRET` | Yes | Secret for Chainlit session encryption |