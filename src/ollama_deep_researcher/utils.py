import os
import glob
import httpx
import requests
from typing import Dict, Any, List, Union, Optional
import logging
from datetime import datetime

from markdownify import markdownify
from langsmith import traceable
from tavily import TavilyClient
from duckduckgo_search import DDGS

from langchain_community.utilities import SearxSearchWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Add logger for local RAG 
logger = logging.getLogger(__name__)

# Function to set up logging with fresh timestamps for each session
def setup_logging():
    """Sets up logging with fresh timestamps for each research session"""
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clean up any existing handlers to prevent duplicate logs
    while root_logger.handlers:
        root_logger.handlers.pop()
    
    # Set up logging to files
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up log levels
    root_logger.setLevel(logging.DEBUG) # Set root logger level

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')  # Simpler format for console
    
    # Main log file handler
    main_log_file = os.path.join(log_dir, f"research_{timestamp}.log") # More generic name
    file_handler = logging.FileHandler(main_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING) # Only show WARNING and above on console
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # --- Silence frequent LangGraph queue logs ---
    try:
        lg_queue_logger = logging.getLogger("langgraph_runtime_inmem.queue")
        lg_queue_logger.setLevel(logging.WARNING) # Only show WARNING and above for this specific logger
        
        # Also silence verbose DEBUG logs from ops logger
        lg_ops_logger = logging.getLogger("langgraph_runtime_inmem.ops")
        lg_ops_logger.setLevel(logging.INFO) # Only show INFO and above for this specific logger
        
    except Exception as e:
        root_logger.warning(f"Could not find or set level for langgraph_runtime_inmem loggers: {e}")
    # --- End silencing ---
    
    root_logger.info("Logging initialized with timestamp: %s - Check logs directory for detailed logs", timestamp)
    return timestamp

def get_config_value(value: Any) -> str:
    """
    Convert configuration values to string format, handling both string and enum types.
    
    Args:
        value (Any): The configuration value to process. Can be a string or an Enum.
    
    Returns:
        str: The string representation of the value.
        
    Examples:
        >>> get_config_value("tavily")
        'tavily'
        >>> get_config_value(SearchAPI.TAVILY)
        'tavily'
    """
    return value if isinstance(value, str) else value.value

def strip_thinking_tokens(text: str) -> str:
    """
    Remove <think> and </think> tags and their content from the text.
    
    Iteratively removes all occurrences of content enclosed in matching thinking tokens.
    If only <think> tag exists without </think>, preserve the entire content unchanged.
    
    Args:
        text (str): The text to process
        
    Returns:
        str: The text with thinking tokens and their content removed if both tags exist
    """
    # Only remove content when both opening and closing tags exist
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    
    # If only <think> tag exists without a matching </think> tag, 
    # preserve the entire content unchanged - this is essential for reflection outputs
    
    return text

def deduplicate_and_format_sources(
    search_response: Union[Dict[str, Any], List[Dict[str, Any]]], 
    max_tokens_per_source: int, 
    fetch_full_page: bool = False
) -> str:
    """
    Format and deduplicate search responses from various search APIs.
    
    Takes either a single search response or list of responses from search APIs,
    deduplicates them by URL, and formats them into a structured string.
    
    Args:
        search_response (Union[Dict[str, Any], List[Dict[str, Any]]]): Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
        max_tokens_per_source (int): Maximum number of tokens to include for each source's content
        fetch_full_page (bool, optional): Whether to include the full page content. Defaults to False.
            
    Returns:
        str: Formatted string with deduplicated sources
        
    Raises:
        ValueError: If input is neither a dict with 'results' key nor a list of search results
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source: {source['title']}\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if fetch_full_page:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sources(search_results: Dict[str, Any]) -> str:
    """
    Format search results into a bullet-point list of sources with URLs.
    
    Creates a simple bulleted list of search results with title and URL for each source.
    
    Args:
        search_results (Dict[str, Any]): Search response containing a 'results' key with
                                        a list of search result objects
        
    Returns:
        str: Formatted string with sources as bullet points in the format "* title : url"
    """
    return '\n'.join(
        f"* {source['title']} : {source['url']}"
        for source in search_results['results']
    )

def fetch_raw_content(url: str) -> Optional[str]:
    """
    Fetch HTML content from a URL and convert it to markdown format.
    
    Uses a 10-second timeout to avoid hanging on slow sites or large pages.
    
    Args:
        url (str): The URL to fetch content from
        
    Returns:
        Optional[str]: The fetched content converted to markdown if successful,
                      None if any error occurs during fetching or conversion
    """
    try:                
        # Create a client with reasonable timeout
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return markdownify(response.text)
    except Exception as e:
        print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")
        return None

@traceable
def duckduckgo_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search the web using DuckDuckGo and return formatted results.
    
    Uses the DDGS library to perform web searches through DuckDuckGo.
    
    Args:
        query (str): The search query to execute
        max_results (int, optional): Maximum number of results to return. Defaults to 3.
        fetch_full_page (bool, optional): Whether to fetch full page content from result URLs. 
                                         Defaults to False.
    Returns:
        Dict[str, List[Dict[str, Any]]]: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str or None): Full page content if fetch_full_page is True,
                                            otherwise same as content
    """
    try:
        with DDGS() as ddgs:
            results = []
            search_results = list(ddgs.text(query, max_results=max_results))
            
            for r in search_results:
                url = r.get('href')
                title = r.get('title')
                content = r.get('body')
                
                if not all([url, title, content]):
                    print(f"Warning: Incomplete result from DuckDuckGo: {r}")
                    continue

                raw_content = content
                if fetch_full_page:
                    raw_content = fetch_raw_content(url)
                
                # Add result to list
                result = {
                    "title": title,
                    "url": url,
                    "content": content,
                    "raw_content": raw_content
                }
                results.append(result)
            
            return {"results": results}
    except Exception as e:
        print(f"Error in DuckDuckGo search: {str(e)}")
        print(f"Full error details: {type(e).__name__}")
        return {"results": []}

@traceable
def searxng_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search the web using SearXNG and return formatted results.
    
    Uses the SearxSearchWrapper to perform searches through a SearXNG instance.
    The SearXNG host URL is read from the SEARXNG_URL environment variable
    or defaults to http://localhost:8888.
    
    Args:
        query (str): The search query to execute
        max_results (int, optional): Maximum number of results to return. Defaults to 3.
        fetch_full_page (bool, optional): Whether to fetch full page content from result URLs.
                                         Defaults to False.
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str or None): Full page content if fetch_full_page is True,
                                           otherwise same as content
    """
    host=os.environ.get("SEARXNG_URL", "http://localhost:8888")
    s = SearxSearchWrapper(searx_host=host)

    results = []
    search_results = s.results(query, num_results=max_results)
    for r in search_results:
        url = r.get('link')
        title = r.get('title')
        content = r.get('snippet')
        
        if not all([url, title, content]):
            print(f"Warning: Incomplete result from SearXNG: {r}")
            continue

        raw_content = content
        if fetch_full_page:
            raw_content = fetch_raw_content(url)
        
        # Add result to list
        result = {
            "title": title,
            "url": url,
            "content": content,
            "raw_content": raw_content
        }
        results.append(result)
    return {"results": results}
    
@traceable
def tavily_search(query: str, fetch_full_page: bool = True, max_results: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search the web using the Tavily API and return formatted results.
    
    Uses the TavilyClient to perform searches. Tavily API key must be configured
    in the environment.
    
    Args:
        query (str): The search query to execute
        fetch_full_page (bool, optional): Whether to include raw content from sources.
                                         Defaults to True.
        max_results (int, optional): Maximum number of results to return. Defaults to 3.
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str or None): Full content of the page if available and 
                                            fetch_full_page is True
    """
     
    tavily_client = TavilyClient()
    return tavily_client.search(query, 
                         max_results=max_results, 
                         include_raw_content=fetch_full_page)

@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int = 0) -> Dict[str, Any]:
    """
    Search the web using the Perplexity API and return formatted results.
    
    Uses the Perplexity API to perform searches with the 'sonar-pro' model.
    Requires a PERPLEXITY_API_KEY environment variable to be set.
    
    Args:
        query (str): The search query to execute
        perplexity_search_loop_count (int, optional): The loop step for perplexity search
                                                     (used for source labeling). Defaults to 0.
  
    Returns:
        Dict[str, Any]: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result (includes search counter)
                - url (str): URL of the citation source
                - content (str): Content of the response or reference to main content
                - raw_content (str or None): Full content for the first source, None for additional
                                            citation sources
                                            
    Raises:
        requests.exceptions.HTTPError: If the API request fails
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "Search the web and provide factual information with sources."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()  # Raise exception for bad status codes
    
    # Parse the response
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Perplexity returns a list of citations for a single search result
    citations = data.get("citations", ["https://perplexity.ai"])
    
    # Return first citation with full content, others just as references
    results = [{
        "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
        "url": citations[0],
        "content": content,
        "raw_content": content
    }]
    
    # Add additional citations without duplicating content
    for i, citation in enumerate(citations[1:], start=2):
        results.append({
            "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
            "url": citation,
            "content": "See above for full content",
            "raw_content": None
        })
    
    return {"results": results}
def query_local_vector_store(query: str, vector_store_paths: List[str], embedding_model: str, limit: int = 5) -> Dict[str, Any]:
    """
    Query local vector stores for relevant documents based on a search query.
    
    Args:
        query (str): The search query to use for retrieval
        vector_store_paths (List[str]): Paths to vector stores to query
        embedding_model (str): Name of the embedding model to use
        limit (int, optional): Number of documents to retrieve. Defaults to 5.
        
    Returns:
        Dict[str, Any]: Search results in a format compatible with web search results
    """
    try:
        # Initialize embedding model
        device = _get_best_available_device()
        logger.info(f"Using device: {device} for embeddings in local RAG")
        logger.info(f"Querying with embedding model: {embedding_model}")
        logger.info(f"Query: {query}")
        logger.info(f"Vector store paths: {vector_store_paths}")
        
        # Try available embedding models in order of preference
        embedding_models_to_try = [
            embedding_model,             # First try the specified model
            "BAAI/bge-m3",               # Then try the model used for preprocessing
            "sentence-transformers/all-MiniLM-L6-v2",  # Standard fallback
            "all-MiniLM-L6-v2"           # Simple name fallback
        ]
        
        # Will store successful embeddings
        embeddings = None
        
        # Try each embedding model
        for model_name in embedding_models_to_try:
            try:
                logger.info(f"Trying embedding model: {model_name}")
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': device}
                )
                # If we're here, the model loaded successfully
                logger.info(f"Successfully loaded embedding model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load embedding model {model_name}: {str(e)}")
        
        if embeddings is None:
            logger.error("Failed to load any embedding model")
            return {"results": [], "source": "local_vector_store"}
        
        all_docs = []
        
        # To avoid "too many open files" error, process batches of vector stores
        batch_size = 20  # Process 20 vector stores at a time
        for i in range(0, len(vector_store_paths), batch_size):
            batch_paths = vector_store_paths[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(vector_store_paths) + batch_size - 1)//batch_size} with {len(batch_paths)} vector stores")
            
            # Process each vector store in the batch
            for store_path in batch_paths:
                if os.path.exists(store_path):
                    logger.info(f"Querying vector store at: {store_path}")
                    
                    # Check vector store structure
                    chroma_files = os.listdir(store_path)
                    logger.info(f"Vector store contains files: {chroma_files}")
                    
                    # Check for SQLite file which indicates Chroma DB
                    if 'chroma.sqlite3' in chroma_files:
                        logger.info("Found SQLite-based Chroma DB format")
                        
                        try:
                            # Try to load the vector store
                            db = Chroma(
                                persist_directory=store_path,
                                embedding_function=embeddings
                            )
                            
                            try:
                                # First check if the vector store has any documents
                                try:
                                    collection_count = db._collection.count()
                                    logger.info(f"Vector store contains {collection_count} documents")
                                    
                                    if collection_count == 0:
                                        logger.warning(f"Vector store {store_path} is empty")
                                        if hasattr(db, '_client') and db._client:
                                            db._client.close()
                                        continue
                                except Exception as e:
                                    logger.warning(f"Could not get document count, will try query anyway: {str(e)}")
                                
                                # Try alternative methods if the database looks valid but can't be queried
                                try:
                                    # Query for similar documents
                                    logger.info(f"Searching for similar documents with limit={limit}")
                                    results = db.similarity_search_with_score(query, k=limit)
                                    
                                    logger.info(f"Found {len(results)} similar documents")
                                    
                                    # Log a sample of the results
                                    if results:
                                        sample_doc, sample_score = results[0]
                                        logger.info(f"Sample result - Score: {sample_score}")
                                        logger.info(f"Sample content: {sample_doc.page_content[:100]}...")
                                        logger.info(f"Sample metadata: {sample_doc.metadata}")
                                    
                                    # Add to overall results
                                    all_docs.extend(results)
                                    
                                except Exception as query_err:
                                    logger.error(f"Error in similarity search: {str(query_err)}")
                                    
                                    # Try direct collection access as fallback
                                    try:
                                        logger.info("Trying direct collection query as fallback")
                                        from chromadb.utils import embedding_functions
                                        
                                        # Get the collection
                                        collection = db._collection
                                        
                                        # Query directly
                                        query_results = collection.query(
                                            query_texts=[query],
                                            n_results=limit
                                        )
                                        
                                        if query_results and 'documents' in query_results:
                                            logger.info(f"Direct query returned {len(query_results['documents'][0])} results")
                                            
                                            # Create Documents from results
                                            for i, doc_text in enumerate(query_results['documents'][0]):
                                                metadata = query_results['metadatas'][0][i] if 'metadatas' in query_results else {}
                                                score = query_results['distances'][0][i] if 'distances' in query_results else 1.0
                                                
                                                doc = Document(page_content=doc_text, metadata=metadata)
                                                all_docs.append((doc, score))
                                    except Exception as direct_err:
                                        logger.error(f"Direct collection query failed: {str(direct_err)}")
                            finally:
                                # Close the database connection to prevent file descriptor leaks
                                if hasattr(db, '_client') and db._client:
                                    try:
                                        db._client.close()
                                        logger.info(f"Closed database connection for {store_path}")
                                    except Exception as close_err:
                                        logger.warning(f"Error closing database connection: {str(close_err)}")
                        
                        except Exception as e:
                            logger.error(f"Error loading or querying vector store {store_path}: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                    else:
                        logger.warning(f"Directory {store_path} doesn't appear to contain a Chroma database")
                else:
                    logger.warning(f"Vector store path not found: {store_path}")
        
        logger.info(f"Total documents found across all stores: {len(all_docs)}")
        
        if not all_docs:
            logger.warning("No documents retrieved from any vector store")
            return {"results": [], "source": "local_vector_store"}
        
        # Sort by similarity score (lower is better)
        all_docs.sort(key=lambda x: x[1])
        
        # Take top 'limit' documents
        top_docs = all_docs[:limit]
        logger.info(f"Selected top {len(top_docs)} documents")
        
        # Format results to match web search format
        results = []
        for doc, score in top_docs:
            # Extract metadata
            metadata = doc.metadata
            
            # --- More robust metadata extraction ---
            # 1. Get Primary ID (Try multiple keys)
            primary_id = metadata.get('paperID', metadata.get('id', metadata.get('chunk_id', 'Unknown')))
            
            # 2. Get Title (Try 'title', fallback to ID)
            title = metadata.get('title', f"Local Document: {primary_id}")
            
            # 3. Get URL (Try 'url', then 'doi', then construct from source/line)
            url = metadata.get('url')
            if not url:
                doi = metadata.get('doi')
                if doi and isinstance(doi, str) and doi != 'Unknown':
                    url = f"https://doi.org/{doi}"
                else:
                    source_file = metadata.get('source_csv', metadata.get('source_jsonl'))
                    line_num = metadata.get('row_number', metadata.get('line_number'))
                    if source_file and line_num:
                        url = f"local://{os.path.basename(source_file)}#L{line_num}" # Use base name
                    else:
                        url = f"local://{primary_id}" # Fallback using primary ID
            # --- End of robust extraction ---

            # Create result structure
            result = {
                "title": title, # Use extracted/generated title
                "url": url,     # Use extracted/generated URL
                "content": doc.page_content[:500] + "...",  # Snippet
                "raw_content": doc.page_content,  # Full content
                "metadata": metadata,  # Include full metadata
                "score": score  # Include similarity score for debugging
            }
            results.append(result)
        
        # Return in format compatible with web search results
        formatted_results = {
            "results": results,
            "source": "local_vector_store"
        }
        
        logger.info(f"Returning {len(results)} formatted results")
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error in local RAG: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Return empty results on error
        return {"results": [], "source": "local_vector_store"}

def _get_best_available_device() -> str:
    """Get the best available device (CUDA GPU, MPS, or CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            # Use the GPU specified by CUDA_VISIBLE_DEVICES
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except:
        return "cpu"

def find_volume_paths(base_path: str) -> List[str]:
    """
    Find all volume directories (vol_*) in the given base path.
    
    Args:
        base_path (str): Base directory to search in
        
    Returns:
        List[str]: List of full paths to volume directories
    """
    # Handle the case where a specific volume is provided
    if os.path.basename(base_path).startswith("vol_") and os.path.isdir(base_path):
        logger.info(f"Using specific volume: {base_path}")
        return [base_path]
    
    # Find all vol_* directories
    volume_pattern = os.path.join(base_path, "vol_*")
    volumes = glob.glob(volume_pattern)
    
    # Filter to only include directories
    volume_dirs = [vol for vol in volumes if os.path.isdir(vol)]
    
    if not volume_dirs:
        logger.warning(f"No volume directories found in {base_path}")
        # Return the original path as fallback
        return [base_path]
    
    logger.info(f"Found {len(volume_dirs)} volume directories: {volume_dirs}")
    return volume_dirs

# Test function for strip_thinking_tokens
def test_strip_thinking_tokens():
    """Test the strip_thinking_tokens function with various cases."""
    # Case 1: Matching tags - should remove tags and content
    test1 = "Before <think>thinking content</think> After"
    result1 = strip_thinking_tokens(test1)
    expected1 = "Before  After"
    print(f"Test 1 - Matching tags:\nResult: '{result1}'\nExpected: '{expected1}'\nPassed: {result1 == expected1}")
    
    # Case 2: Multiple matching tags - should remove all tags and content
    test2 = "Start <think>thinking 1</think> Middle <think>thinking 2</think> End"
    result2 = strip_thinking_tokens(test2)
    expected2 = "Start  Middle  End"
    print(f"Test 2 - Multiple tags:\nResult: '{result2}'\nExpected: '{expected2}'\nPassed: {result2 == expected2}")
    
    # Case 3: Only opening tag - should preserve ENTIRE content unchanged
    test3 = "Before <think>thinking content without closing tag"
    result3 = strip_thinking_tokens(test3)
    expected3 = "Before <think>thinking content without closing tag"
    print(f"Test 3 - Only opening tag:\nResult: '{result3}'\nExpected: '{expected3}'\nPassed: {result3 == expected3}")
    
    # Case 4: Real-world example with matching tags
    test4 = "Here's a summary of the research:\n\n<think>\nLet me organize my thoughts about what I've found...\n</think>\n\nThe analysis shows that..."
    result4 = strip_thinking_tokens(test4)
    expected4 = "Here's a summary of the research:\n\n\nThe analysis shows that..."
    print(f"Test 4 - Real-world example with matching tags:\nResult: '{result4}'\nExpected: '{expected4}'\nPassed: {result4 == expected4}")
    
    # Case 5: Real-world example with only opening tag - preserve exactly as is
    test5 = "Here's a summary of the research:\n\n<think>\nLet me organize my thoughts about what I've found...\n\nThe analysis shows that..."
    result5 = strip_thinking_tokens(test5)
    expected5 = "Here's a summary of the research:\n\n<think>\nLet me organize my thoughts about what I've found...\n\nThe analysis shows that..."
    print(f"Test 5 - Real-world example with only opening tag:\nResult: '{result5}'\nExpected: '{expected5}'\nPassed: {result5 == expected5}")
    
    # Case 6: Only thinking content without any other text
    test6 = "<think>this is just thinking process"
    result6 = strip_thinking_tokens(test6)
    expected6 = "<think>this is just thinking process"
    print(f"Test 6 - Only thinking content:\nResult: '{result6}'\nExpected: '{expected6}'\nPassed: {result6 == expected6}")

if __name__ == "__main__":
    # Run the test function
    test_strip_thinking_tokens()