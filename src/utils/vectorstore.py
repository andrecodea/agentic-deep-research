"""Initializes and exposes the Chroma vector store for the research agent.

Here I used a singleton pattern to ensure that the vector store is initialized only once.
"""

import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vector_store: Chroma | None = None # empty var (will be initialized later)

def get_vector_store() -> Chroma:
    """Returns the singleton Chroma vector store instance."""
    try:
        global _vector_store # persistence across calls (modifies global var inside func)
        if _vector_store is None: # first call?
            _vector_store = Chroma( # initializes vstore instance
                collection_name="research_agent",
                embedding_function=embeddings,
                persist_directory="./chroma_db"
            )
            log.info("Chroma vector store initialized - Collection: 'research_agent'")
        return _vector_store # any call -> returns instance
    except Exception as e:
        log.error(f"Error initializing Chroma vector store: {e}", exc_info=True)
        raise