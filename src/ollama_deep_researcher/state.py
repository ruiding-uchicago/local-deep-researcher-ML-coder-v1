import operator
import uuid
from dataclasses import dataclass, field
from typing_extensions import Annotated

@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None) # Report topic     
    search_query: str = field(default=None) # Search query
    complementary_search_query: str = field(default=None) # Complementary search query based on local RAG results
    web_research_results: Annotated[list, operator.add] = field(default_factory=list) 
    complementary_web_research_results: Annotated[list, operator.add] = field(default_factory=list) 
    local_research_results: Annotated[list, operator.add] = field(default_factory=list) 
    local_rag_summary: str = field(default=None) # Summary of local RAG resultss
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list) 
    complementary_sources_gathered: Annotated[list, operator.add] = field(default_factory=list) 
    local_sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0) # Research loop count
    running_summary: str = field(default=None) # Final report
    summary_history: Annotated[list, operator.add] = field(default_factory=list) # History of summaries from each iteration
    
    # Add unique ID for each research node
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Add status tracking for research processing
    # Possible values: "pending", "in_progress", "completed", "analyzed"
    processing_status: str = field(default="pending")

@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None) # Report topic     

@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None) # Final report

# --- State for Code Improvement Workflow ---

@dataclass(kw_only=True)
class CodeImprovementInput:
    problem_description: str = field(default=None, metadata={"title": "Problem Description", "description": "Describe the overall ML task or problem."})
    current_code: str = field(default=None, metadata={"title": "Current Code", "description": "Paste the current version of the code."})
    current_performance: str = field(default=None, metadata={"title": "Current Performance", "description": "Describe the code's current performance (metrics, issues)."})
    human_request: str = field(default=None, metadata={"title": "Specific Request", "description": "Specific instructions or areas to focus on for improvement."})

@dataclass(kw_only=True)
class CodeImprovementState:
    # Inputs
    problem_description: str = field(default=None)
    current_code: str = field(default=None)
    current_performance: str = field(default=None)
    human_request: str = field(default=None)
    
    # Intermediate results
    critique: str = field(default=None) # Added: Critique from analysis node
    research_topic: str = field(default=None) 
    research_report: str = field(default=None) 
    revision_plan: str = field(default=None) 
    
    # Final output
    revised_code: str = field(default=None) 
    
    # Keep track of original input for reference
    original_input: CodeImprovementInput = field(default=None)
    
    # Error tracking (optional)
    error_message: str = field(default=None)

@dataclass(kw_only=True)
class CodeImprovementOutput:
    revised_code: str = field(default=None)
    revision_plan: str = field(default=None)