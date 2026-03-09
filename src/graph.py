"""Graph construction for the Deep Research Graph System."""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, START, END

from utils.state import ResearchState
from utils.nodes import (
    orchestrator,
    retriever,
    researcher,
    writer,
)

log = logging.getLogger(__name__)


def route_intent(state: ResearchState) -> str:
    """Routes the graph based on the user's intent classified by Orchestrator."""
    intent = state.get("intent", "conversation")

    if intent == "conversation":
        return "writer"
    elif intent == "simple_search":
        return "retriever"
    elif intent == "quick_search":
        return "researcher"
    elif intent == "deep_research":
        # Deep research must always hit internal knowledge first to check cache
        return "retriever" 
    
    return "writer"


def route_after_retriever(state: ResearchState) -> str:
    """Decides if we need web search after checking the vector database."""
    intent = state.get("intent", "")
    cache_hit = state.get("cache_hit", False)

    # If it's deep research and internal docs weren't enough, go to web
    if intent == "deep_research" and not cache_hit:
        return "researcher"

    # Otherwise, go straight to writing the report
    return "writer"


def build_graph(checkpointer) -> StateGraph:
    """
    Builds and compiles the Deep Research StateGraph.
    
    Args:
        checkpointer: The initialized saver (e.g., PostgresSaver, MemorySaver) 
                      passed from the main execution script.
    """
    graph = StateGraph(ResearchState)

    # -- Nodes --
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("retriever", retriever)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)

    # -- Edges --
    graph.add_edge(START, "orchestrator")

    # Orchestrator -> branch by intent
    # LangGraph automatically maps the returned string to the node name if they match.
    graph.add_conditional_edges(
        "orchestrator",
        route_intent,
        ["writer", "retriever", "researcher"]
    )

    # Retriever -> branch to web search OR writer
    graph.add_conditional_edges(
        "retriever",
        route_after_retriever,
        ["researcher", "writer"]
    )

    # Researcher always goes to Writer
    graph.add_edge("researcher", "writer")

    # Writer to END
    graph.add_edge("writer", END)

    # Compile the graph
    return graph.compile(checkpointer=checkpointer)

log.info("Graph built successfully.")