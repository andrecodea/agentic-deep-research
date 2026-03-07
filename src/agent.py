"""Graph construction fo the Deep Research Graph System."""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from utils.state import ResearchState
from utils.nodes import (
    orchestrator,
    router,
    retriever,
    researcher,
    upserter,
    writer,
    hitl,
)

log = logging.getLogger(__name__)


def route_intent(state: ResearchState) -> str:
    intent = state.get("intent")
    cache_hit = state.get("cache_hit", False)

    match intent:
        case "conversation":
            return "writer"
        case "simple_search":
            return "retriever"
        case "quick_search":
            return "researcher"
        case "deep_research":
            return "writer" if cache_hit else "researcher"
        case _:
            return "writer"


def route_after_retriever(state: ResearchState) -> str:
    intent = state.get("intent", "")
    cache_hit = state.get("cache_hit", False)

    if intent == "deep_research" and not cache_hit:
        return "researcher"

    return "writer"


def should_save(state: ResearchState) -> str:
    if state.get("save_to_chroma"):
        return "upserter"
    return END


def build_graph() -> StateGraph:
    db_url = os.environ["DATABASE_URL"]
    graph = StateGraph(ResearchState)

    # -- Nodes --
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("router", router)
    graph.add_node("retriever", retriever)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("hitl", hitl)
    graph.add_node("upserter", upserter)

    # -- Edges --
    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "router")

    # Router -> branch by intent
    graph.add_conditional_edges(
        "router",
        route_intent,
        {
            "writer": "writer",
            "retriever": "retriever",
            "researcher": "researcher",
        },
    )

    # Retriever -> writer or researhcer (deep research cache miss)
    graph.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "researcher": "researcher",
            "writer": "writer",
        },
    )

    # HITL -> upserter or END
    graph.add_conditional_edges(
        "hitl",
        should_save,
        {
            "upserter": "upserter",
            END: END,
        },
    )

    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "hitl")

    graph.add_edge("upserter", END)
    with PostgresSaver.from_conn_string(db_url) as checkpointer:
        checkpointer.setup()
        return graph.compile(checkpointer=checkpointer, interrupt_before=["hitl"])


log.info("Graph built successfully.")
