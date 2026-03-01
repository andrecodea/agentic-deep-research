# AGENTS.md - Deep Research Agent

## Project Overview

Enterprise-grade multi-agent research pipeline orchestrated with LangGraph, designed to accelerate technical decision-making by automating web research, knowledge curation, and report generation.

- **Stack**: LangGraph (StateGraph), Mercury 2 (Inception Labs), Tavily (search/extract), Chroma (vector store), LangSmith (observability), PostgreSQL (session persistence)
- **Python**: 3.12+
- **Package Manager**: uv

## Build / Run / Test Commands

### Install Dependencies
```bash
uv sync
```

### Run Locally
```bash
langgraph dev
```

### Run with Docker
```bash
docker compose up
```

### Run Tests
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest path/to/test.py    # Run specific file
pytest path/to/test.py::test_function_name  # Run single test
pytest -k "test_name"     # Run tests matching pattern
```

### Lint / Format
```bash
ruff check .              # Lint all files
ruff check . --fix        # Lint and auto-fix
ruff format .             # Format code
```

## Code Style Guidelines

### Imports

Order imports in each file:
1. Standard library (`logging`, `typing`, `sys`, etc.)
2. Third-party packages (`langchain`, `dotenv`, `tavily`, etc.)
3. Local application (`from utils.state import...`)

```python
import logging
from typing import List, Literal, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from utils.state import ResearchState
from utils.prompts import get_prompt
```

### Docstrings

Use Google-style docstrings. Include Args and Returns sections for all public functions.

```python
def add_docs(existing: list, new: list) -> list:
    """Reducer that appends a new document to an existing one.

    Works for both list[Document] (vstore retrieval) and list[dict] (vstore upsert).

    Args:
        existing: Existing documents.
        new: New documents to append.

    Returns:
        Updated list of documents.
    """
    return existing + new
```

### Type Hints

- Use `Literal` for enum-like string values (e.g., `"research" | "conversation"`)
- Use `Annotated` with reducers for state fields that accumulate values
- Use explicit type hints for all function parameters and return values

```python
class ResearchState(MessagesState):
    intent: Literal["research", "conversation"]
    research_mode: Literal["new", "existing"]
    query: str
    retrieved_docs: Annotated[list[Document], add_docs]  # reducer for accumulated docs
    cache_hit: bool
    final_report: str
    save_to_chroma: bool
```

### Naming Conventions

- **Functions/variables**: snake_case (`orchestrator`, `search_results`)
- **Classes**: PascalCase (`ResearchState`, `TavilyClient`)
- **Constants**: SCREAMING_SNAKE_CASE (if needed)
- **Files**: snake_case (`state.py`, `tools.py`)

### Error Handling

Always wrap external API calls in try/except, log the error, and re-raise.

```python
def tavily_search(query: str, topic: str) -> list[dict]:
    try:
        response = tavily_client.search(...)
        return response["results"]
    except Exception as e:
        log.error(f"Error searching the web: {e}", exc_info=True)
        raise
```

### Logging

Initialize a logger in every module using the module's `__name__`.

```python
import logging

log = logging.getLogger(__name__)
```

### Retry Logic

Use tenacity decorators for external API calls (Tavily, Chroma, LLMs).

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException, ConnectionError))
)
def tavily_search(query: str, topic: str) -> list[dict]:
    ...
```

### State Management

- Extend `MessagesState` to inherit message handling with default append reducer
- Use custom reducers (`add_docs`) when accumulating rather than replacing values
- Keep state minimal: only include fields needed by downstream nodes

### LangGraph Nodes

- Each node is a function that takes `state: ResearchState` and returns a `dict`
- Return dict contains only the fields being updated
- Use `...` (Ellipsis) for stub/placeholder nodes pending implementation

```python
def orchestrator(state: ResearchState) -> dict:
    chain = ORCHESTRATOR_PROMPT | llm
    try:
        response = chain.invoke({"messages": state["messages"]})
        return {"intent": response.content, "query": state["query"]}
    except Exception as e:
        log.error(f"Failed to get response: {e}", exc_info=True)
        raise

def router(state: ResearchState) -> dict:
    ...
```

## Environment Configuration

Create a `.env` file in the project root:

```bash
# LLM
INCEPTION_API_KEY=your_key_here

# Search
TAVILY_API_KEY=your_key_here

# Observability
LANGSMITH_API_KEY=your_key_here

# Session persistence
POSTGRES_URI=postgresql://user:pass@localhost:5432/db
```

## Project Structure

```
deep_research_agent/
├── src/
│   ├── agent.py              # StateGraph compilation + entry point
│   └── utils/
│       ├── state.py          # ResearchState (shared state + reducers)
│       ├── nodes.py          # orchestrator, router, retriever, researcher, writer, hitl
│       ├── tools.py          # Tavily search/extract, Chroma query/save
│       ├── prompts.py        # LangSmith Hub prompt retrieval
│       └── vectorstore.py    # Chroma singleton initialization
├── tests/                    # pytest unit tests (add your tests here)
├── .env                      # Environment variables
├── pyproject.toml            # Project dependencies
└── langgraph.json            # LangGraph configuration
```

## Testing Guidelines

When writing tests:
- Use pytest as the test framework
- Name test files as `test_*.py`
- Place tests in the `tests/` directory
- Use descriptive test names: `test_orchestrator_classifies_research_intent`
- Mock external API calls where possible
- Test state reducers independently

## Adding New Nodes

1. Define the node function in `src/utils/nodes.py`
2. Add the field(s) to `ResearchState` in `src/utils/state.py` if needed
3. Add the node to the graph in `src/agent.py` using `graph.add_node()`
4. Define edges in the graph compilation
