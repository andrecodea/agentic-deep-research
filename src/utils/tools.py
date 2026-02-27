"""Manages the graph's tools with retry clauses.

tavily_search: web search using the tavily web search API.
tavily_extract: content extraction using the tavily web search API.
vector_store_upsert: Chroma knowledge base document upsertion.
vector_store_retrieval: Chroma knowledge base document retrieval.
"""

import logging  
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool
from utils.vectorstore import get_vector_store
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx


load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

tavily_client = TavilyClient()

@tool
@retry(
    stop=stop_after_attempt(3), # 3 attempts
    wait=wait_exponential(multiplier=1, min=2, max=10), # wait 2s before 2nd attempt, wait 4s before 3rd attempt, raise after 3rd failures
    retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException, ConnectionError))
)
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

# TODO: TEST
@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException, ConnectionError))
)
def tavily_extract(urls: list[str] | str) -> list[dict]:
    """Extract full content from the given url in Markdown format.

    Args:
        urls (list[str] | str): URLs to extract content from.

    Returns:
        list[dict]: List of extracted results with url, raw_content and images
    
    """
    try:
        response = tavily_client.extract(
            urls=urls,
            include_images=True, # extracts imgs for md summary
            format="markdown"
        )
        results = response["results"]
        log.info(f"Tavily extract completed - URLs: {len(results)} | extracted: {len(results)}")
        return results
    except Exception as e:
        log.error(f"Error extracting data from pages: {e}", exc_info=True)
        raise

# TODO: REVIEW -> TEST
@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def vector_store_upsert(results: list[dict]) -> str:
    """Embeds and upserts documents into a Chroma knowledge base.

    Args:
        results (list[dict]): Extraction results from tavily_extract containing,
        raw_content, url and images.

    Returns:
        str: Confirmation message with number of documents upserted.
    """
    try:
        vector_store = get_vector_store() # lazy init
        vector_store.add_texts(
            texts=[result["raw_content"] for result in results],
            metadatas=[{
                "source": result["url"],
                "images": result.get("images", [])
            } for result in results]
        )
        log.info(f"{len(results)} documents upserted successfully.")
        return f"{len(results)} documents upserted into the knowledge base succesfully."
    except Exception as e:
        log.error(f"Failed to upsert documents {e}", exc_info=True)
        raise

# TODO: REVIEW -> TEST
@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def vector_store_retrieval(query: str) -> list[tuple]:
    """Retrieves documents from a Chroma knowledge base

    Args:
        query (str): text query for retrieval (e.g. "attention is all you need paper")

    Returns
        list[tuple[Document, float]]: List of tuples containing the documents
        and their respective scores, the lower, the better.
    """
    try:
        vector_store = get_vector_store()
        results: list[tuple] = vector_store.similarity_search_with_score(query, k=3)
        log.info(f"{len(results)} documents retrieved successfully.")
        return results
    except Exception as e:
        log.error(f"Failed to retrieve documents {e}", exc_info=True)
        raise
