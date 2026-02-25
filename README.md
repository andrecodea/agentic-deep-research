# Deep Research Multi-Agent System with RAG via HITL

> Multi-agent research pipeline with semantic memory, 
> HITL curation, and LangGraph orchestration.

## Architecture
```
  Initial State:
  ├── messages: []
  ├── retrieved_docs: []
  └── search_results: 
  Orchestrator writes:
  └── intent: "researc
  Router writes:
  └── research_mode: "ne
  Retriever writes:
  ├── query: "LangGraph tutorial"
  ├── retrieved_docs: ["doc from cache"]   ← add_docs accumulates
  └── cache_hit: Fal
  Researcher writes:
  └── search_results: ["search result from Tavily"]  ← add_docs accumulates
  Writer writes:
  └── final_report: "# Report..
  HITL writes:
  └── save_to_chroma: True
```

## Stack
- **Orchestration**: LangGraph StateGraph
- **LLM**: GPT-5.2 Thinking
- **Search**: Tavily (search + extract)
- **Memory**: Chroma (semantic cache via HITL)
- **Observability**: LangSmith
- **Interface**: Agent Chat UI