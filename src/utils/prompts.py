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

def get_prompt(prompt_name: str, version: str = None) -> str:
    """Retrieves ChatPromptTemplate from LangSmith Hub.
    
    Args
        prompt_name (str): prompt name on LangSmith Hub 
        version (str): prompt version on LangSmith Hub
    
    Example
        ```python
        from prompts import get_prompt
        prompt = get_prompt("username/prompt-name", "v0.0.1")
        prompt.invoke({"query": "What is the capital of France?"})
        ```
    
    Returns
        A LangChain ChatPromptTemplate object containing the system prompt.
    """
    try:
        prompt: ChatPromptTemplate = client.pull_prompt(
            prompt_name,
            include_model=False,
            version=version # pulls latest automatically if version is set to None
        )

        # Returned object has the 'messages' attribute? If so, is it empty?
        if hasattr(prompt, 'messages') and len(prompt.messages) > 0:

            # If not, get the first message
            msg = prompt.messages[0]

            # Check if the message has the 'prompt' attribute (ChatPromptTemplate)
            if hasattr(msg, 'prompt'):
                system_prompt = msg.prompt.template

            # Check if the message has the 'content' attribute (HumanMessage/SystemMessage)
            elif hasattr(msg, 'content'):
                system_prompt = msg.content
            else:
                raise ValueError(f"Prompt '{prompt_name}' returned an unexpected structure.")
            
            log.info(f"Successfully retrieved prompt '{prompt_name}' version {version}")
            return system_prompt

        else:
            raise ValueError(f"Prompt '{prompt_name}' returned an empty message list.")
    except Exception as e:
        log.error(f"Error retrieving prompt: {e}", exc_info=True)
        raise