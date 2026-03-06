"""Graph nodes for the Deep Research multi-agent system."""

import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser

from utils.state import ResearchState
from utils.prompts import get_prompt
from utils.tools import tavily_extract, tavily_search, vector_store_retrieval, vector_store_upsert

load_dotenv()
log = logging.getLogger(__name__)

# Prompt pulling
ORCHESTRATOR_PROMPT = get_prompt("deep-research-orchestrator:latest")
RESEARCHER_PROMPT = get_prompt("deep-research-researcher:latest")
WRITER_PROMPT = get_prompt("deep-research-writer:latest")

# Global llm
llm = ChatOpenAI(
    model="mercury-2",
    temperature=0.0,
    base_url="https://api.inceptionlabs.ai/v1"
)

def orchestrator(state: ResearchState) -> dict:
    """Initializes orchestrator agent with ResearchState.

    The orchestrator classifies intent and extracts research query from user messages.
    
    Args:
        state (ResearchState): State of the agent, defines the communication protocol between nodes.
    
    Returns:
        dict: User `intent` and user `query` to be passed on to the router node.
    """
    
    log.info("Orchestrator executing...")

    prompt = ORCHESTRATOR_PROMPT
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    try:
        response = chain.invoke({"messages": state["messages"]})

        # Fallback: if response is not in dict format
        if not isinstance(response, dict):
            log.warning(f"Unexpected response type: {type(response)}")
            return {"intent": "conversation", "query": state.get("query", "")}
        
        intent = response.get("intent", "conversation")
        query = response.get("query") or  state.get("query", "")
        
        log.info(f"Orchestrator completed: intent={intent}, query={query[:50]}...")
        return {"intent": intent, "query": query}
    except Exception as e:
        log.error(f"Failed to get response: {e}",exc_info=True)
        raise

def router(state: ResearchState) -> dict:
    """Initializes router with ResearchState.
    
    - If intent != research: default to "existing" (no web serach needed)
    - If intent == research: default to "new" (web search activated)
    """
    log.info(f"Router executing: intent={state.get('intent')}")
    
    intent = state.get("intent")

    if intent != "research":
        log.info("Router: non-research intent, defaulting to 'existing' mode")
        return {"research_mode": "existing"}
    
    log.info("Router: research intent, defaulting to new")
    return {"research_mode": "new"}

def retriever(state: ResearchState) -> dict:
    """Retrieves documents from a ChromaDB vectorstore through similarity search.
    
    Args:
        state (ResearchState): Current graph state.
            query (ResearchState.query): User query for retrieval.
    
    Returns:
        dict: Retrieved documents and cache hit status.
    """
    log.info(f"Retriever executing: query={state.get('query')}")
    query = state.get("query", "")

    try:
        results = vector_store_retrieval(query)

        if not results:
            log.info("Retriever: no documents found.")
            return {"retrieved_docs": [], "cache_hit": False}
        
        best_score = results[0][1] # list[tuple[Document, float]] e.g. [(doc, simmilarity score)]
        cache_hit = best_score < 0.3 # True if best score above 0.3
        retrieved_docs = [doc for doc, _ in results]
        
        log.info(f"Retriever: score={best_score:.3f}, cache_hit={cache_hit}")
        return {"retrieved_docs": retrieved_docs, "cache_hit": cache_hit}
    except Exception as e:
        log.error(f"Retriever failed: {e}", exc_info=True)
        raise
        

def researcher(state: ResearchState) -> dict:
    """Conduct web search and page content extraction via Tavily.

    Workflow:
        1. Gets query from state
        2. Searches the web with the given query
        3. Extracts content from the web pages of the searches based in their URLs

    Args:
        state (ResearchState): Current graph state.
            query (ResearchState.query): A user's search query.
    
    Returns
        dict: Search results
    """
    log.info(f"Researcher executing: query={state.get('query')}")
    try:
        query = state.get("query", "")

        search_results = tavily_search(query)

        if not search_results:
            log.warning(f"Researcher: no results found for '{query}'")
            return {"search_results": []}

        log.info(f"Researcher: {len(search_results)} results found.")

        urls = [result["url"] for result in search_results]
        extracted = tavily_extract(urls)

        if not extracted:
            log.warning(f"Researcher: extraction of {urls} returned no content.")
            return {"search_results": []}

        log.info(f"Researcher: {len(extracted)} pages extracted.")
        return {"search_results": {extracted}}
    except Exception as e:
        log.error(f"Researcher failed {e}", exc_info=True)
        raise

def upserter(state: ResearchState) -> dict:
    """Upserts documents into a ChromaDB vectorstore.
    
    Args:
        state (ResearchState): Current graph state.
            search_results (ResearchState.search_results): Extracted web content
    Returns:
        dict: Empty - state unchanged.
    """
    log.info(f"Upserter executing: search_results={state.get('search_results')}")
    
    try:
        search_results = state.get("search_results", [])

        if not search_results:
            log.info("Retriever: no search results to persist.")
            return {}
        
        vector_store_upsert(search_results)
        log.info(f"Upserter: {len(search_results)} documents upserted")
        return {}
    except Exception as e:
        log.error(f"Upserter failed: {e}", exc_info=True)
        raise
        
#TODO
def writer(state: ResearchState) -> dict:
    ...

# TODO
def hitl(state: ResearchState) -> dict:
    ...