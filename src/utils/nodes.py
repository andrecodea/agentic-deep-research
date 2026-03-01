"""Graph nodes for the Deep Research multi-agent system."""

import logging
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from utils.state import ResearchState
from utils.prompts import get_prompt
from utils.tools import *

load_dotenv()
log = logging.getLogger(__name__)

# Prompt pulling
ORCHESTRATOR_PROMPT = get_prompt("deep-research-orchestrator:latest")
ROUTER_PROMPT = get_prompt("deep-research-router:latest")
RETRIEVER_PROMPT = get_prompt("deep-research-retriever:latest")
RESEARCHER_PROMPT = get_prompt("deep-research-researcher:latest")
WRITER_PROMPT = get_prompt("deep-research-writer:latest")

# Global llm
llm = ChatOpenAI(
    model="mercury-2",
    temperature=0.5,
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

    chain = ORCHESTRATOR_PROMPT | llm | JsonOutputParser()
    
    try:
        response = chain.invoke({"messages": state["messages"]})
        content = response.content

        # Fallback: if response is not in dict format
        if not isinstance(response, dict):
            log.warning(f"Unexpected response type: {type(response)}")
            return {
                "intent": "conversation", 
                "query": state.get("query", "")}
        
        log.info(f"Orchestrator completed: intent={intent}, query={query[:50]}...")
        return {
            "intent": response.get("intent", "conversation"), 
            "query": response.get("query", state.get("query", ""))
            }
    except Exception as e:
        log.error(f"Failed to get response: {e}",exc_info=True)
        raise

# TODO
def router(state: ResearchState) -> dict:
    ...

# TODO
def retriever(state: ResearchState) -> dict:
    ...

# TODO
def researcher(state: ResearchState) -> dict:
    ...

#TODO
def writer(state: ResearchState) -> dict:
    ...

# TODO
def hitl(state: ResearchState) -> dict:
    ...