"""Graph nodes for the Deep Research multi-agent system."""

import logging
from dotenv import load_dotenv
from typing import Union

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver, RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt as hitl_interrupt
from langchain.agents.middleware import HumanInTheLoopMiddleware


from utils.prompts import get_prompt
from utils.state import ResearchState
from utils.tools import (
    tavily_extract,
    tavily_search,
    vector_store_retrieval,
    vector_store_upsert,
)
from utils.vectorstore import get_vector_store

import os
load_dotenv()
log = logging.getLogger(__name__)

# Inline prompts
ORCHESTRATOR_PROMPT = get_prompt("andrecodea/deep-research-orchestrator")

WRITER_PROMPT = get_prompt("andrecodea/deep-research-writer")

checkpointer = InMemorySaver()

# Global llm
llm = ChatOpenAI(
    model="mercury-2",
    temperature=0.0,
    streaming=True,
    base_url="https://api.inceptionlabs.ai/v1",
    api_key=os.getenv("INCEPTION_API_KEY")
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

        if not isinstance(response, dict):
            log.warning(f"Unexpected response type: {type(response)}")
            return {"intent": "conversation", "query": state.get("query", ""), "topic": state.get("topic", "")}

        intent = response.get("intent", "conversation")
        query = response.get("query") or state.get("query", "")
        topic = response.get("topic", "")

        log.info(f"Orchestrator completed: intent={intent}, query={query[:50]}...")
        return {"intent": intent, "query": query, "topic": topic}
    except Exception as e:
        log.error(f"Failed to get response: {e}", exc_info=True)
        raise


def router(state: ResearchState) -> dict:
    """Initializes router with ResearchState.

    - If intent != research: default to "existing" (no web serach needed)
    - If intent == research: default to "new" (web search activated)
    """
    log.info(f"Router executing: intent={state.get('intent')}")

    intent = state.get("intent")

    if intent == "deep_research":
        return {"research_mode": "new"}

    if intent == "simple_search":
        return {"research_mode": "existing"}

    if intent == "quick_search":
        return {"research_mode": "quick"}

    log.info("Router: research intent, defaulting to new")
    return {"research_mode": "existing"}


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

        best_score = results[0][1]
        cache_hit = best_score < 0.3
        retrieved_docs = [doc for doc, _ in results]

        log.info(f"Retriever: score={best_score:.3f}, cache_hit={cache_hit}")
        return {"retrieved_docs": retrieved_docs, "cache_hit": cache_hit}
    except Exception as e:
        log.error(f"Retriever failed: {e}", exc_info=True)
        raise


def researcher(state: ResearchState) -> dict:
    """Conduct web search and page content extraction via Tavily.

    Args:
        state (ResearchState): Current graph state.
            query (ResearchState.query): A user's search query.
            topic (ResearchState.topic): The search's topic

    Returns
        dict: Search results
    """
    log.info(f"Researcher executing: query={state.get('query')}")
    try:
        query = state.get("query", "")
        topic = state.get("topic", "")
        research_mode = state.get("research_mode", "")
        search_results = tavily_search(query, topic)

        if not search_results:
            log.warning(f"Researcher: no results found for '{query}'")
            return {"search_results": []}

        if research_mode == "quick":
            log.info(f"Researcher: {len(search_results)} results found.")
            return {"search_results": search_results}

        if research_mode == "new":
            urls = [result["url"] for result in search_results]
            extracted = tavily_extract(urls)
            if not extracted:
                log.warning(f"Researcher: extraction of {urls} returned no content.")
                return {"search_results": []}

            log.info(f"Researcher: {len(extracted)} pages extracted.")
            return {"search_results": extracted}
    except Exception as e:
        log.error(f"Researcher failed: {e}", exc_info=True)
        raise


# Agent creation
writer_agent = create_agent(
    model=llm,
    system_prompt=WRITER_PROMPT.messages[0].prompt.template,
    tools=[vector_store_upsert],
    checkpointer=checkpointer,
    state_schema=ResearchState,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "vector_store_upsert": True, 
            },
            description_prefix="Salvar pesquisa no ChromaDB?",
        ),
    ],)

async def writer(state: ResearchState, config: RunnableConfig) -> dict:
    """Generates final report using the agentic LLM with context.

    Args:
        state (ResearchState): Current graph state.

    Returns:
        dict: final_report with generated content.
    """
    log.info("Writer: executing")
    intent = state.get("intent", "")

    try:
        if intent in ["simple_search", "quick_search", "deep_research"]:
            docs_text = "\n\n".join(
                [doc.page_content for doc in state.get("retrieved_docs", [])]
            )

            results_text = "\n\n".join(
                [
                    f"Source: {r.get('url', '')}\n{r.get('raw_content') or r.get('content', '')}"
                    for r in state.get("search_results", [])]
            )

            context_prompt = (
                f"Escreva o relatório final baseado na seguinte pesquisa:\n\n"
                f"Sua tarefa (Query): {state.get('query', '')}\n\n"
                f"--- DOCUMENTOS DA BASE ---\n{docs_text}\n\n"
                f"--- RESULTADOS DA WEB ---\n{results_text}"
            )
            context_message = HumanMessage(content=context_prompt)

            response = await writer_agent.ainvoke(
                {"messages": [context_message]},
                config
            )
        else:
            response = await writer_agent.ainvoke(
                {"messages": [state["messages"][-1]]},
                config
            )
        ai_message = response["messages"][-1]
        final_report = ai_message.content
        return {
            'messages': [ai_message],
            'final_report': final_report
            }
    except Exception as e:
        log.error(f"Writer failed: {e}", exc_info=True)
        raise