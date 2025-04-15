import os
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Literal, List, Union

from langchain_core.runnables import RunnableConfig
defaul_config_long_recursion = RunnableConfig(recursion_limit=300)

###
class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"

class Configuration(BaseModel):
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = Field(
        default=2,
        title="Research Depth",
        description="Number of research iterations to perform"
    )
    local_llm: str = Field(
        default="deepseek-r1:14b",
        title="LLM Model Name",
        description="Name of the LLM model to use"
    )
    llm_provider: Literal["ollama", "lmstudio"] = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider for the LLM (Ollama or LMStudio)"
    )
    search_api: Literal["perplexity", "tavily", "duckduckgo", "searxng"] = Field(
        default="duckduckgo",
        title="Search API",
        description="Web search API to use"
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/",
        title="Ollama Base URL",
        description="Base URL for Ollama API"
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API"
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses"
    )
    use_local_rag: bool = Field(
        default=True,
        title="Use Local RAG",
        description="Whether to use local vector store for RAG before web search"
    )
    vector_store_paths: List[str] = Field(
        default=["/Users/ruiding/mac_python_folder/langchain_agent/vector_store_cs/"],
        title="Vector Store Paths",
        description="Paths to local vector stores, comma-separated if multiple"
    )
    local_results_count: int = Field(
        default=5,
        title="Local Results Count",
        description="Number of chunks to retrieve from local vector store"
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        title="Embedding Model",
        description="Embedding model to use for vector store queries"
    )
    
    @model_validator(mode='before')
    @classmethod
    def parse_vector_store_paths(cls, data: Any) -> Any:
        """Parse vector_store_paths from string to list if needed."""
        if isinstance(data, dict) and 'vector_store_paths' in data:
            # If it's a string, split by comma to make a list
            if isinstance(data['vector_store_paths'], str):
                data['vector_store_paths'] = [
                    path.strip() for path in data['vector_store_paths'].split(',')
                ]
        return data
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[defaul_config_long_recursion] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }
        
        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}
        
        return cls(**values)