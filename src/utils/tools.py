import logging  
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool
from vectorstore import get_vector_store

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

tavily_client = TavilyClient()

@tool
def tavily_search(query: str, topic: str) -> list[dict]:
    """Searches the web for the given query.

    Args:
        query (str): Query to search the web for.
        topic (str): Topic context ("general", "news", "finance")

    Returns:
        list[dict]: List of the top 3 search results with URL, title and snippet.
    """
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            topic=topic,
            max_results=3,
            include_images=False
        )
        
        search_results = response["results"]
        log.info(f"Tavily search completed — query: '{query}' | results: {len(search_results)}")
        return search_results
    except Exception as e:
        log.error(f"Error searching the web: {e}", exc_info=True)
        raise

# TODO
@tool
def tavily_extract():
    pass

# TODO
@tool
def vector_store_upsert():
    pass

# TODO
@tool
def vector_store_retrieval():
    pass
