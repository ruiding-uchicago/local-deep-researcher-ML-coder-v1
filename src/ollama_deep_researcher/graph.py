import json
import os
import random
import logging
import re
from datetime import datetime
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from ollama_deep_researcher.configuration import Configuration, SearchAPI
from ollama_deep_researcher.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search, duckduckgo_search, searxng_search, strip_thinking_tokens, get_config_value, query_local_vector_store, find_volume_paths, setup_logging
from ollama_deep_researcher.state import SummaryState, SummaryStateInput, SummaryStateOutput
from ollama_deep_researcher.prompts import query_writer_instructions, complementary_query_writer_instructions, summarizer_instructions, reflection_instructions, get_current_date
from ollama_deep_researcher.lmstudio import ChatLMStudio

# Define a logger for this module
logger = logging.getLogger(__name__)

defaul_config_long_recursion = RunnableConfig(recursion_limit=300)

# Helper function to initialize a new session
def init_session(state: SummaryStateInput, config: RunnableConfig = None) -> SummaryStateInput:
    """Initialize a new research session, including creating fresh log files."""
    # Set up fresh logging for this session
    timestamp = setup_logging()
    
    # Return the input state unchanged
    return state

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """LangGraph node that generates a search query based on the research topic.
    
    Uses an LLM to create an optimized search query for web research based on
    the user's research topic. Supports both LMStudio and Ollama as LLM providers.
    
    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            client_kwargs={"timeout": 6000.0},
            format="json"
        )
    else: # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            client_kwargs={"timeout": 6000.0},
            format="json"
        )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Generate a query for web search:")]
    )
    
    # Get the content
    content = result.content

    # Parse the JSON response and get the query
    try:
        query = json.loads(content)
        search_query = query['query']
    except (json.JSONDecodeError, KeyError):
        # If parsing fails or the key is not found, use a fallback query
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
        search_query = content
    return {"search_query": search_query}

def local_rag_research(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs local RAG (Retrieval Augmented Generation) using the search query.
    
    Queries local vector stores for relevant documents based on the search query.
    This is done before any web search to leverage existing knowledge.
    
    Args:
        state: Current graph state containing the search query
        config: Configuration for the runnable, including vector store paths and settings
        
    Returns:
        Dictionary with state update, including local_research_results and local_sources_gathered
    """
    # Configure
    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)
    
    # Skip if use_local_rag is false
    if not configurable.use_local_rag:
        return {"local_research_results": [], "local_sources_gathered": []}
    
    # Get vector store paths
    vector_paths = []
    
    # Process each configured path to find all volumes
    for base_path in configurable.vector_store_paths:
        # Find all volume directories in this base path
        volume_paths = find_volume_paths(base_path)
        vector_paths.extend(volume_paths)
    
    # Query local vector store
    search_results = query_local_vector_store(
        query=state.search_query,
        vector_store_paths=vector_paths,
        embedding_model=configurable.embedding_model,
        limit=configurable.local_results_count
    )
    
    # Format results
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=2500, fetch_full_page=True)
    local_sources = format_sources(search_results)
    
    return {
        "local_research_results": [search_str],
        "local_sources_gathered": [local_sources]
    }
def generate_complementary_query(state: SummaryState, config: RunnableConfig):
    """LangGraph node that generates a complementary search query based on local RAG results.
    
    Analyzes local knowledge to create a query that explores a different but related angle 
    of the research topic, ensuring diverse search results.
    
    Args:
        state: Current graph state containing the original search query and local RAG results
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including complementary_search_query containing the diversified query
    """
    # If local RAG is disabled or no results, return empty query
    
    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)
    print(f"=== ENTERING generate_complementary_query ===")
    print(f"use_local_rag: {configurable.use_local_rag}, local_research_results (sample): {state.local_research_results[0]}")
    print(f"type: {type(state.local_research_results)}, is empty: {not state.local_research_results}")
    
    # Debug the conditional evaluation directly
    condition_part1 = not configurable.use_local_rag
    condition_part2 = not state.local_research_results
    print(f"condition_part1 (not use_local_rag): {condition_part1}")
    print(f"condition_part2 (not local_research_results): {condition_part2}")
    print(f"full condition: {condition_part1 or condition_part2}")
    
    if not configurable.use_local_rag or not state.local_research_results:
        # Return empty result, forcing the system to skip this search
        print("Returning empty complementary query!")
        return {"complementary_search_query": ""}
        
    # Format the prompt
    current_date = get_current_date()
    # use local_rag_summary instead of local_research_results 
    local_rag_result = state.local_rag_summary
    
    # Use a simplified prompt format instead of the complex one that might confuse the model
    formatted_prompt = f"""Your goal is to generate a complementary search query based on local knowledge.

<CONTEXT>
Current date: {current_date}
Original research topic: {state.research_topic}
Original search query: {state.search_query}
</CONTEXT>


<GOAL>
Generate a different but related query that explores a new angle on the research topic.
</GOAL>

<FORMAT>
Format your response as a JSON object with this exact key:
   - "query": A search query that explores a different angle of the research topic
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "(think of a different angle on alternative or complementary approaches to) {state.research_topic}"
}}
</EXAMPLE>

Provide your response in JSON format:"""
    
    print(f"Using research_topic: {state.research_topic}")
    print(f"Using original_query: {state.search_query}")
    print(f"Local RAG result length: {len(local_rag_result)}")
    print(f"Formatted prompt first 200 chars: {formatted_prompt[:200]}...")
    print(f"Local RAG summary: {state.local_rag_summary}")

    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        print(f"Using LMStudio with base_url: {configurable.lmstudio_base_url}, model: {configurable.local_llm}")
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0,  # Changed from 0.3 to 0 to match other working nodes
            client_kwargs={"timeout": 6000.0},
            format="json"
        )
    else: # Default to Ollama
        print(f"Using Ollama with base_url: {configurable.ollama_base_url}, model: {configurable.local_llm}")
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0,  # Changed from 0.3 to 0 to match other working nodes
            client_kwargs={"timeout": 6000.0},
            format="json"
        )
    
    print("About to invoke LLM...")
    try:
        # Make the human message explicit about what we want
        result = llm_json_mode.invoke(
            [SystemMessage(content=formatted_prompt),
             HumanMessage(content="Based on the following local RAG summary, <LOCAL_KNOWLEDGE> {state.local_rag_summary}</LOCAL_KNOWLEDGE>. Generate a complementary query (different from the original query {state.search_query}) in JSON format with ONLY a 'query' field:")]
        )
        print("LLM invocation completed")
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        # Fallback if LLM fails, randomly pick one from the list
        fallback_queries = [f"Alternative perspectives on {state.research_topic}", f"Different angles on {state.research_topic}", f"New perspectives on {state.research_topic}", f"Complementary approaches to {state.research_topic}"]
        return {"complementary_search_query": random.choice(fallback_queries)}
    
    # Get the content
    content = result.content
    print(f"Raw LLM response: {content[:300]}...")

    # New recursive function to find any query at any level in the JSON
    def find_query_in_json(json_obj):
        """Recursively search for keys containing 'query' in a JSON object at any level."""
        if isinstance(json_obj, dict):
            # First try to find direct keys containing 'query'
            query_keys = [k for k in json_obj.keys() if 'query' in k.lower()]
            if query_keys:
                # Sort by priority - exact match first, then by length (shorter is better)
                query_keys.sort(key=lambda x: (0 if x.lower() == 'query' else 1, len(x)))
                key = query_keys[0]
                if json_obj[key] and isinstance(json_obj[key], str):
                    print(f"Found query under key '{key}': {json_obj[key]}")
                    return json_obj[key]
            
            # If not found, search recursively in nested objects
            for key, value in json_obj.items():
                result = find_query_in_json(value)
                if result:
                    return result
                    
        elif isinstance(json_obj, list):
            # Search in each element of the list
            for item in json_obj:
                result = find_query_in_json(item)
                if result:
                    return result
                    
        return None

    # Parse the JSON response and get the complementary query
    try:
        query_data = json.loads(content)
        print(f"Full parsed JSON: {query_data}")
        
        # First try the recursive search for any query at any level
        complementary_query = find_query_in_json(query_data)
        
        # If not found through recursive search, try the original method as fallback
        if not complementary_query:
            print("Recursive search failed, trying original method")
            # Try to get the query from the 'query' key which we asked for in our prompt
            if 'query' in query_data and query_data['query']:
                complementary_query = query_data['query']
                print(f"Found query: {complementary_query}")
            else:
                # Try other possible keys as fallback
                possible_keys = ['complementary_query', 'search_query', 'follow_up_query', 'query_to_search', 'new_query','complementary_query_to_search']
                complementary_query = ""
                
                for key in possible_keys:
                    if key in query_data and query_data[key]:
                        complementary_query = query_data[key]
                        print(f"Found query under alternative key '{key}': {complementary_query}")
                        break
        
        # If still empty, try to use the entire content as the query if it looks like plain text
        if not complementary_query and not content.startswith('{') and not content.startswith('['):
            print("Using raw content as query")
            complementary_query = content.strip()
        
        # Last resort fallback
        if not complementary_query:
            print("Empty complementary query generated, using fallback")
            complementary_query = f"Alternative aspects of {state.research_topic}"
            
        print(f"Final complementary query: '{complementary_query}'")
        
        # Check if the complementary query is too similar to the original query
        if complementary_query.lower().strip() == state.search_query.lower().strip():
            print("Warning: Complementary query is identical to original query. Trying backup approach...")
            
            # Direct backup approach without using local RAG content
            backup_prompt = f"""Generate a complementary search query for the research topic.

<ORIGINAL_INFORMATION>
Research topic: {state.research_topic}
Original search query: {state.search_query}
</ORIGINAL_INFORMATION>

<INSTRUCTIONS>
1. You MUST generate a query that is DIFFERENT from the original query.
2. Take a completely different perspective or angle on the research topic.
3. Format your response as a JSON object with a single "query" field.
4. Keep your query concise and focused.
5. DO NOT return the original query or anything too similar.
</INSTRUCTIONS>

Example output:
{{
    "query": "limitations and criticisms of {state.research_topic}"
}}

Provide only the JSON response:"""

            try:
                # Set higher temperature for more creativity in the backup attempt
                if configurable.llm_provider == "lmstudio":
                    backup_llm = ChatLMStudio(
                        base_url=configurable.lmstudio_base_url, 
                        model=configurable.local_llm, 
                        temperature=0,  
                        client_kwargs={"timeout": 6000.0},
                        format="json"
                    )
                else:
                    backup_llm = ChatOllama(
                        base_url=configurable.ollama_base_url, 
                        model=configurable.local_llm, 
                        temperature=0,  
                        client_kwargs={"timeout": 6000.0},
                        format="json"
                    )
                
                backup_result = backup_llm.invoke(
                    [SystemMessage(content=backup_prompt),
                     HumanMessage(content=f"Generate a complementary query that is DIFFERENT from: '{state.search_query}'")]
                )
                
                backup_content = backup_result.content
                print(f"Backup approach result: {backup_content[:300]}...")
                
                try:
                    backup_data = json.loads(backup_content)
                    backup_query = find_query_in_json(backup_data)
                    
                    if backup_query and backup_query.lower().strip() != state.search_query.lower().strip():
                        print(f"Using backup query: {backup_query}")
                        return {"complementary_search_query": backup_query}
                    else:
                        # If backup also failed or returned the same query, use a template fallback, randomly pick one from the list
                        fallback_queries = [f"Criticisms and limitations of {state.research_topic}", f"Alternative perspectives on {state.research_topic}", f"Different angles on {state.research_topic}", f"New perspectives on {state.research_topic}", f"Complementary approaches to {state.research_topic}"]
                        print("Backup approach failed or returned same query, using template fallback")
                        return {"complementary_search_query": random.choice(fallback_queries)}
                        
                except (json.JSONDecodeError, KeyError):
                    # If JSON parsing fails, use raw content if it doesn't look like JSON
                    if not backup_content.startswith('{') and not backup_content.startswith('['):
                        if backup_content.lower().strip() != state.search_query.lower().strip():
                            return {"complementary_search_query": backup_content.strip()}
                    
                    # Last resort template fallback, randomly pick one from the list
                    fallback_queries = [f"Alternative approaches to {state.research_topic}", f"Different approaches to {state.research_topic}", f"New approaches to {state.research_topic}", f"Complementary approaches to {state.research_topic}"]
                    return {"complementary_search_query": random.choice(fallback_queries)}
                    
            except Exception as e:
                print(f"Error during backup LLM invocation: {e}")
                # Last resort template fallback, randomly pick one from the list
                fallback_queries = [f"Disadvantages and limitations of {state.research_topic}", f"Criticisms and limitations of {state.research_topic}", f"Alternative perspectives on {state.research_topic}", f"Different angles on {state.research_topic}", f"New perspectives on {state.research_topic}", f"Complementary approaches to {state.research_topic}"]
                return {"complementary_search_query": random.choice(fallback_queries)}
        
        return {"complementary_search_query": complementary_query}
    except (json.JSONDecodeError, KeyError) as e:
        # If parsing fails, return empty query instead of using a fallback
        print(f"JSON parsing failed with error: {e}")
        print(f"Raw content: {content[:200]}...")
        
        # Try to use the raw content if it's not JSON
        if not content.startswith('{') and not content.startswith('['):
            print("Using raw content as query since it's not JSON")
            return {"complementary_search_query": content.strip()}
            
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
            print(f"After stripping tokens: {content[:200]}...")
            
        print("All attempts failed - returning fallback query")
        # Last resort template fallback, randomly pick one from the list
        fallback_queries = [f"Different perspectives on {state.research_topic}", f"Alternative perspectives on {state.research_topic}", f"Different angles on {state.research_topic}", f"New perspectives on {state.research_topic}", f"Complementary approaches to {state.research_topic}"]
        return {"complementary_search_query": random.choice(fallback_queries)}
def web_research(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs web research using the generated search query.
    
    Executes a web search using the configured search API (tavily, perplexity, 
    duckduckgo, or searxng) and formats the results for further processing.
    
    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings
        
    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """

    # Configure
    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)

    # Get the search API
    search_api = get_config_value(configurable.search_api)

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(state.search_query, fetch_full_page=configurable.fetch_full_page, max_results=3)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=2500, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=2500, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(state.search_query, max_results=5, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=2500, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "searxng":
        search_results = searxng_search(state.search_query, max_results=5, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=2500, fetch_full_page=configurable.fetch_full_page)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def complementary_web_research(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs web research using the complementary search query.
    
    Similar to the web_research node but uses the complementary search query that explores
    a different but related angle of the research topic.
    
    Args:
        state: Current graph state containing the complementary search query
        config: Configuration for the runnable, including search API settings
        
    Returns:
        Dictionary with state update, including complementary_sources_gathered and complementary_web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)

    # Get the search API
    search_api = get_config_value(configurable.search_api)

    # If complementary query is not available, skip this node
    if not state.complementary_search_query:
        return {"complementary_sources_gathered": [], "complementary_web_research_results": []}

    # Search the web with complementary query
    if search_api == "tavily":
        search_results = tavily_search(state.complementary_search_query, fetch_full_page=configurable.fetch_full_page, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=2500, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.complementary_search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=2500, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(state.complementary_search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=2500, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "searxng":
        search_results = searxng_search(state.complementary_search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=2500, fetch_full_page=configurable.fetch_full_page)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {
        "complementary_sources_gathered": [format_sources(search_results)], 
        "complementary_web_research_results": [search_str]
    }

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """LangGraph node that summarizes research results from both local and web sources.
    
    Uses an LLM to create or update a running summary based on the newest research 
    results, integrating them with any existing summary.
    
    Args:
        state: Current graph state containing research topic, running summary,
              and research results from both local and web sources
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including running_summary key containing the updated summary
    """

    # Existing summary
    existing_summary = state.running_summary

    # Get the most recent research results
    research_content = ""
    
    # Add local research results if available
    if state.local_research_results and len(state.local_research_results) > 0:
        research_content += f"<Local Research Results>\n{state.local_research_results[-1]}\n</Local Research Results>\n\n"
    
    # Add web research results from original query if available
    if state.web_research_results and len(state.web_research_results) > 0:
        research_content += f"<Main Research Results>\n{state.web_research_results[-1]}\n</Main Research Results>\n\n"
    
    # Add complementary web research results if available
    if state.complementary_web_research_results and len(state.complementary_web_research_results) > 0:
        research_content += f"<Complementary Research Results>\n{state.complementary_web_research_results[-1]}\n</Complementary Research Results>\n\n"

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {research_content} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {research_content} \n <Context>"
            f"Create a Summary using the Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0.3, #we cannot use temperature 0 because the model would stuck in a loop repeating the same summary over and over
            client_kwargs={"timeout": 6000.0} # Add timeout via client_kwargs
        )
    else:  # Default to Ollama
        llm = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0.3, #we cannot use temperature 0 because the model would stuck in a loop repeating the same summary over and over
            client_kwargs={"timeout": 6000.0} # Add timeout via client_kwargs
        )
    
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    # Strip thinking tokens if configured
    running_summary = result.content
    if configurable.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)
    
    # Add current summary to summary_history
    # Create an object with metadata about this summary iteration
    iteration_summary = {
        "iteration": state.research_loop_count,
        "summary": running_summary,
        "query": state.search_query,
        "complementary_query": state.complementary_search_query if state.complementary_search_query else ""
    }
    
    # Return updated state with the new summary and updated history
    return {
        "running_summary": running_summary,
        "summary_history": [iteration_summary]
    }

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """LangGraph node that identifies knowledge gaps and generates follow-up queries.
    
    Analyzes the current summary to identify areas for further research and generates
    a new search query to address those gaps. Uses structured output to extract
    the follow-up query in JSON format.
    
    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """

    # Log the input state
    logger.info(f"===== REFLECTING ON SUMMARY - INPUT STATE =====")
    logger.info(f"Research topic: {state.research_topic}")
    logger.info(f"Research loop count: {state.research_loop_count}")
    logger.info(f"Summary length: {len(state.running_summary) if state.running_summary else 0} chars")
    
    # Generate a query
    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            client_kwargs={"timeout": 6000.0},
            format="json"
        )
    else: # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            client_kwargs={"timeout": 6000.0},
            format="json"
        )
    
    logger.info("Sending reflection prompt to LLM...")
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n And now identify a knowledge gap and generate a follow-up web search query:")]
    )
    #debug why there is no follow up query, print raw LLM response 
    logger.info(f"Raw LLM response: {result.content}")
    
    # Strip thinking tokens if configured
    try:
        # Try to parse as JSON first
        reflection_content = json.loads(result.content)
        # Get the follow-up query
        query = reflection_content.get('follow_up_query')
        knowledge_gap = reflection_content.get('knowledge_gap', 'No knowledge gap specified')
        
        logger.info(f"Knowledge gap identified: {knowledge_gap}")
        
        # Check if query is None or empty
        if not query:
            # Use a fallback query, randomly pick one from the list
            fallback_queries = [f"Tell me more about {state.research_topic}", f"What are the latest trends in {state.research_topic}?", f"What are the latest developments in {state.research_topic}?", f"What are the latest advancements in {state.research_topic}?", f"What are the latest innovations in {state.research_topic}?", f"What are the latest breakthroughs in {state.research_topic}?", f"What are the latest advancements in {state.research_topic}?", f"What are the latest innovations in {state.research_topic}?", f"What are the latest breakthroughs in {state.research_topic}?"]
            query = random.choice(fallback_queries)
            logger.info(f"Using fallback query: {query}")
        
        logger.info(f"Generated follow-up query: {query}")
        logger.info("===== REFLECTION COMPLETE =====")
        return {"search_query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails or the key is not found, use a fallback query
        logger.warning("Failed to parse LLM response as JSON, using fallback query")
        fallback_queries = [f"Tell me more about {state.research_topic}", f"What are the latest trends in {state.research_topic}?", f"What are the latest developments in {state.research_topic}?", f"What are the latest advancements in {state.research_topic}?", f"What are the latest innovations in {state.research_topic}?", f"What are the latest breakthroughs in {state.research_topic}?", f"What are the latest advancements in {state.research_topic}?", f"What are the latest innovations in {state.research_topic}?", f"What are the latest breakthroughs in {state.research_topic}?"]
        query = random.choice(fallback_queries)
        logger.info(f"Using fallback query: {query}")
        logger.info("===== REFLECTION COMPLETE =====")
        return {"search_query": query}
        
def finalize_summary(state: SummaryState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.
    
    Uses an LLM to create a comprehensive final summary by analyzing multiple iterations 
    of research summaries. Then combines this with deduplicated sources to create a 
    well-structured research report with proper citations.
    
    Args:
        state: Current graph state containing the running summary, summary history, and sources gathered
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    # Configure LLM
    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)
    
    # Log the input state
    logger.info(f"===== FINALIZING SUMMARY - INPUT STATE =====")
    logger.info(f"Research topic: {state.research_topic}")
    logger.info(f"Research loop count: {state.research_loop_count}")
    logger.info(f"Summary history entries: {len(state.summary_history) if state.summary_history else 0}")
    logger.info(f"Sources gathered: {len(state.sources_gathered) if state.sources_gathered else 0}")
    logger.info(f"Complementary sources: {len(state.complementary_sources_gathered) if state.complementary_sources_gathered else 0}")
    logger.info(f"Local sources: {len(state.local_sources_gathered) if state.local_sources_gathered else 0}")
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0.3, #we cannot use temperature 0 because the model would stuck in a loop repeating the same summary over and over
            client_kwargs={"timeout": 6000.0} # Add timeout via client_kwargs
        )
    else:  # Default to Ollama
        llm = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0.3, #we cannot use temperature 0 because the model would stuck in a loop repeating the same summary over and over
            client_kwargs={"timeout": 6000.0} # Add timeout via client_kwargs
        )
    
    # Process summary history for use in the final synthesis
    # If we have summary history, we'll use that; otherwise we'll use the running_summary
    summary_history_text = ""
    
    if state.summary_history and len(state.summary_history) > 0:
        # We have summary history, format it for the LLM
        logger.info("Formatting summary history for final synthesis")
        for i, summary_item in enumerate(state.summary_history):
            summary_history_text += f"\n--- ITERATION {summary_item.get('iteration', i+1)} ---\n"
            summary_history_text += f"Search Query: {summary_item.get('query', 'Unknown')}\n"
            if summary_item.get('complementary_query'):
                summary_history_text += f"Complementary Query: {summary_item.get('complementary_query')}\n"
            summary_history_text += f"Summary:\n{summary_item.get('summary', '')}\n\n"
    else:
        # No history available, just use the current summary
        logger.info("No summary history available, using current running summary")
        summary_history_text = f"Only one research iteration was performed. Summary:\n{state.running_summary}"
    
    # Build the prompt
    final_summary_prompt = f"""
    <GOAL>
    Create a comprehensive final research report on: {state.research_topic}
    </GOAL>

    <CONTEXT>
    You have conducted {state.research_loop_count} iterations of research on this topic.
    Below you'll find summaries from each research iteration.
    </CONTEXT>

    <RESEARCH_HISTORY>
    {summary_history_text}
    </RESEARCH_HISTORY>

    <INSTRUCTIONS>
    1. Create a well-structured, comprehensive final report that synthesizes all the research findings across iterations
    2. Organize information logically with clear section headings
    3. Highlight key insights, patterns, and conclusions
    4. Show how the research evolved across iterations, noting how later iterations built upon or shifted from earlier findings
    5. Note any remaining gaps or areas for future research
    6. Make the report easy to read and understand for someone unfamiliar with the topic
    7. Aim for ~1000-1500 words for the main content (not including sources)
    8. ALWAYS provide a complete summary, even if complex. If you're thinking through the process, make sure to finish with your final report.
    </INSTRUCTIONS>
    """
    
    # Get the final summary from the LLM
    logger.info("Generating final summary...")
    result = llm.invoke(
        [SystemMessage(content=final_summary_prompt),
         HumanMessage(content=f"The User has queried for a comprehensive research on \n<User Input>\n {state.research_topic} \n <User Input>. Based on the <RESEARCH_HISTORY>, please create the final comprehensive research report on: {state.research_topic}")]
    )
    
    # Strip thinking tokens if configured
    final_content = result.content
    if configurable.strip_thinking_tokens:
        final_content = strip_thinking_tokens(final_content)
    
    logger.info(f"Final summary generated: {len(final_content)} chars")
    
    # Deduplicate sources before joining
    logger.info("Processing and deduplicating sources...")
    seen_sources = set()
    unique_sources = []
    
    # Process web sources from original query
    for source in state.sources_gathered:
        # Split the source into lines and process each individually
        for line in source.split('\n'):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)
    
    # Process web sources from complementary query
    for source in state.complementary_sources_gathered:
        # Split the source into lines and process each individually
        for line in source.split('\n'):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append("🔍 " + line)  # Add a magnifying glass emoji to denote complementary search source
    
    # Process local sources
    for source in state.local_sources_gathered:
        # Split the source into lines and process each individually
        for line in source.split('\n'):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append("📚 " + line)  # Add a book emoji to denote local source
    
    # Join the deduplicated sources
    all_sources = "\n".join(unique_sources)
    logger.info(f"Deduplicated sources: {len(unique_sources)} unique sources")
    
    # Combine LLM-generated final summary with sources
    final_report = f"{final_content}\n\n### Sources:\n{all_sources}"
    
    logger.info("===== FINAL SUMMARY COMPLETE =====")
    
    # --- Save final report to file ---
    try:
        # Create the directory if it doesn't exist
        output_dir = "final_summary"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a safe filename from the research topic and timestamp
        safe_topic = re.sub(r'[\/*?"<>|]', "", state.research_topic) # Remove invalid chars
        safe_topic = re.sub(r'\s+', '_', safe_topic) # Replace spaces with underscores
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_topic[:50]}_{timestamp}.md" # Limit topic length in filename
        filepath = os.path.join(output_dir, filename)
        
        # Write the report to the file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(final_report)
        logger.info(f"Final report saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving final report to file: {e}")
    # --- End save final report ---
    
    return {"running_summary": final_report}

def summarize_local_rag_results(state: SummaryState, config: RunnableConfig):
    """LangGraph node that summarizes local RAG results before generating a complementary query.
    
    Creates a concise summary of the local RAG results to help with generating
    a more focused complementary query.
    
    Args:
        state: Current graph state containing the local RAG results
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including local_rag_summary key
    """
    # If local RAG is disabled or no results, return empty summary
    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)
    
    if not configurable.use_local_rag or not state.local_research_results:
        return {"local_rag_summary": ""}
    
    # Use all local RAG results, to be noted, the local_research_results is a list of strings
    # so we need to join them together with two \n to form a single string rather than passing a list
    local_rag_result = "\n\n".join(state.local_research_results)  if state.local_research_results else ""
    #debug
    print(f"Local RAG result: {local_rag_result[:100]}")
    # Build the summary prompt
    summary_prompt = f"""
    <GOAL>
    Create a concise summary (<500 characters) of the local knowledge provided below.
    </GOAL>

    <CONTEXT>
    These are results from a local knowledge base search related to: {state.research_topic}
    Search query used: {state.search_query}
    </CONTEXT>


    <INSTRUCTIONS>
    1. Summarize the key findings, concepts, and information from the local knowledge
    2. Focus on aspects that might suggest different angles to explore in further research
    3. Keep your summary under 500 characters
    4. Start with "After querying local literature with '{state.search_query}', we found that..."
    5. Always provide a complete summary, even if thinking through a complex process.
    </INSTRUCTIONS>
    """
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0.3, #we cannot use temperature 0 because the model would stuck in a loop repeating the same summary over and over
            client_kwargs={"timeout": 6000.0} # Add timeout via client_kwargs
        )
    else:  # Default to Ollama
        llm = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0.3, #we cannot use temperature 0 because the model would stuck in a loop repeating the same summary over and over
            client_kwargs={"timeout": 6000.0} # Add timeout via client_kwargs
        )
    
    # Get the summary from the LLM
    result = llm.invoke(
        [SystemMessage(content=summary_prompt),
         HumanMessage(content=f"Please summarize the local knowledge base results: \n\n {local_rag_result}")]
    )
    
    # Strip thinking tokens if configured
    local_rag_summary = result.content
    if configurable.strip_thinking_tokens:
        local_rag_summary = strip_thinking_tokens(local_rag_summary)
    
    return {"local_rag_summary": local_rag_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "local_rag_research"]:
    """LangGraph routing function that determines the next step in the research flow.
    
    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.
    
    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting
        
    Returns:
        String literal indicating the next node to visit ("local_rag_research" or "finalize_summary")
    """

    configurable = Configuration.from_runnable_config(defaul_config_long_recursion)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "local_rag_research"
    else:
        return "finalize_summary"

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)

# Add the initialization node
builder.add_node("init_session", init_session)
builder.add_node("generate_query", generate_query)
builder.add_node("local_rag_research", local_rag_research)
builder.add_node("summarize_local_rag_results", summarize_local_rag_results)
builder.add_node("generate_complementary_query", generate_complementary_query)
builder.add_node("web_research", web_research)
builder.add_node("complementary_web_research", complementary_web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "init_session")
builder.add_edge("init_session", "generate_query")
builder.add_edge("generate_query", "local_rag_research")
builder.add_edge("local_rag_research", "summarize_local_rag_results")
builder.add_edge("summarize_local_rag_results", "generate_complementary_query")
builder.add_edge("generate_complementary_query", "web_research")
builder.add_edge("web_research", "complementary_web_research")
builder.add_edge("complementary_web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()
# Add step timeout to prevent CancelledError
graph.step_timeout = 86400  # 24 hours in seconds
