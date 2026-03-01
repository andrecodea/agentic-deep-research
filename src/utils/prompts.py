"""Manages the agent's system prompt using LangSmith Hub"""

import logging
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

# Logger setup
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LangSmith client
client = Client()

def get_prompt(prompt_name: str) -> ChatPromptTemplate:
    """Retrieves a `ChatPromptTemplate` object from LangSmith Hub.
    
    Args:
    prompt_name (str): prompt name on LangSmith Hub 
        
    
    Example:
    ```python
    from prompts import get_prompt
    prompt = get_prompt("username/prompt-name:v0.0.1")
    chain = prompt | llm
    chain.invoke({"query": "What is the capital of France?"})
    ```
    
    Returns:
    A LangChain ChatPromptTemplate object containing the system prompt.
    """
    try:
        prompt: ChatPromptTemplate = client.pull_prompt(
            prompt_name,
            include_model=False,
        )
        return prompt
    except Exception as e:
        log.error(f"Error retrieving prompt: {e}", exc_info=True)
        raise