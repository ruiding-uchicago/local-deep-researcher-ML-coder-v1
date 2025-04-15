from datetime import datetime

# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

query_writer_instructions="""Your goal is to generate a targeted web search query.

<CONTEXT>
Current date: {current_date}
Please ensure your queries account for the most current information available as of this date.
</CONTEXT>

<TOPIC>
{research_topic}
</TOPIC>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "machine learning transformer architecture explained",
    "rationale": "Understanding the fundamental structure of transformer models"
}}
</EXAMPLE>

Provide your response in JSON format:"""

complementary_query_writer_instructions="""Your goal is to generate a COMPLEMENTARY web search query that explores a different angle of the research topic.

<CONTEXT>
Current date: {current_date}
Original research topic: {research_topic}
Original search query: {original_query}
</CONTEXT>

<LOCAL_KNOWLEDGE>
{local_rag_results}
</LOCAL_KNOWLEDGE>

<GOAL>
You must carefully analyze the local knowledge provided above to identify areas to explore that are DIFFERENT from but still RELEVANT to the original query.
Rather than enhancing the original query, your task is to create a GENERALLY DIFFERENT query that:
1. Explores an alternative aspect or perspective of the topic
2. Targets information that is NOT present in the local knowledge but would be valuable
3. Complements the original query by investigating related but distinct concepts
4. Maintains relevance to the scientific domain (chemistry, materials science, engineering, etc.)
5. Is GENERALLY DIFFERENT from the original query (avoid just adding words like "advanced" or "technical details")
</GOAL>

<FORMAT>
Format your response as a JSON object with these exact keys:
   - "complementary_query": A search query that explores a different angle of the research topic
   - "divergent_aspect": Explain what different aspect of the topic this query explores
   - "rationale": Why this complementary angle will provide valuable additional information
</FORMAT>

<EXAMPLE>
Example for original query "machine learning transformer architecture explained":
{{
    "complementary_query": "limitations of attention mechanisms in large language models",
    "divergent_aspect": "While the original query focuses on the architecture and structure, this explores the limitations and weaknesses",
    "rationale": "Understanding both the strengths and limitations provides a more complete picture of transformer technology"
}}
</EXAMPLE>

<EXAMPLE>
Example for original query "novel chemical candidates selectively binding PFOA":
{{
    "complementary_query": "alternative remediation technologies for PFAS compounds beyond selective binding",
    "divergent_aspect": "This explores different approaches to PFAS remediation beyond binding mechanisms",
    "rationale": "A comprehensive solution may involve multiple remediation strategies working together"
}}
</EXAMPLE>

Provide your response in JSON format:"""

summarizer_instructions="""
<GOAL>
Generate a high-quality summary of the provided context.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.                                                                                                                                                            
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.
- Always write a complete summary, even if complex. If you're thinking through the process, always conclude with the final summary output.  
< /FORMATTING >

<Task>
Think carefully about the provided Context first. Then generate a summary of the context to address the User Input.
</Task>
"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for web search.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}
</Task>

Provide your analysis in JSON format:"""

# --- Prompts for Code Improvement Workflow ---

analyze_code_and_plan_research_instructions = """You are an expert ML code reviewer and research planner.

<GOAL>
Analyze the provided ML code context (description, code, performance, request) to:
1. Identify key weaknesses, potential bugs, or areas for improvement in the code.
2. Determine the most critical knowledge gaps that need to be addressed through targeted research.
3. Formulate a concise and actionable research topic focused on finding solutions or best practices for the identified issues.
</GOAL>

<INPUT_CONTEXT>
Problem Description: {problem_description}
---
Current Code:
```python
{current_code}
```
---
Current Performance: {current_performance}
---
Human Request: {human_request}
</INPUT_CONTEXT>

<REQUIREMENTS>
- Your critique should be specific and constructive.
- The research topic must directly address the most significant issues identified in the critique.
- The topic should be suitable for web search to find relevant papers, articles, or documentation.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- "critique": A concise summary of the main issues found in the code and context.
- "research_topic": The specific research topic to investigate for solutions.
</FORMAT>

<EXAMPLE>
{{
    "critique": "The current data augmentation pipeline seems basic and might not be robust enough for the described dataset variability. Performance degradation suggests overfitting.",
    "research_topic": "advanced data augmentation techniques for image classification with limited data variability"
}}
</EXAMPLE>

Provide your analysis and research plan in JSON format:"""

generate_revision_plan_instructions = """You are an expert ML engineer responsible for planning code revisions based on research findings.

<GOAL>
Synthesize the original code context and the research report to create a **clear, concise, and actionable** high-level revision plan (under 500 words).
This plan should guide a developer on *what* to change and *why*, based on the research report.
</GOAL>

<ORIGINAL_CONTEXT>
Problem Description: {problem_description}
---
Current Code Snippet (for reference):
```python
{current_code_snippet}
```
---
Current Performance: {current_performance}
---
Human Request: {human_request}
</ORIGINAL_CONTEXT>

<RESEARCH_REPORT>
{research_report}
</RESEARCH_REPORT>

<REQUIREMENTS>
- Refer to specific findings in the research report to justify planned changes.
- Break down the plan into logical steps or areas of modification.
- Focus on *what* needs to change (e.g., "Replace activation function", "Implement early stopping", "Refactor data loading") and *why* (e.g., "based on research findings X", "to address performance issue Y").
- **Keep the plan concise, actionable, and under 500 words.**
- Do NOT write the revised code in this step.
- Do NOT include lengthy explanations, focus on the direct plan.
</REQUIREMENTS>

<FORMAT>
Output the revision plan as **concise** markdown text.
Use headings, bullet points, or numbered lists for clarity.
Start directly with the plan (e.g., with a heading like "Revision Plan").
</FORMAT>

Provide only the concise revision plan in markdown format:"""

implement_code_revision_instructions = """You are an expert Python programmer functioning as a code execution engine.

<GOAL>
Your SOLE task is to meticulously revise the provided `Original Code` based *EXCLUSIVELY* on the instructions in the `Revision Plan`. You MUST NOT deviate from the plan.
</GOAL>

<INPUTS>
--- Original Code ---
```python
{current_code}
```
--- Revision Plan ---
{revision_plan}
--- Context (For understanding the plan only) ---
Problem Description: {problem_description}
Original Performance: {current_performance}
Original Human Request: {human_request}
</INPUTS>

<OUTPUT_REQUIREMENTS>
1.  **Implement ONLY the changes specified in the Revision Plan.** Do not add, remove, or modify anything not explicitly mentioned in the plan.
2.  **Maintain the original code's structure** unless the plan requires refactoring.
3.  **Output ONLY the complete, revised Python code.**
4.  **DO NOT include ANY explanations, comments (unless clarifying a specific planned change), introductory sentences, or concluding remarks.**
5.  **ABSOLUTELY NO '<think>' or reasoning blocks.** Your output MUST start directly with `import` or the first line of the revised code.
6.  The output code MUST be syntactically correct Python.
</OUTPUT_REQUIREMENTS>

Directly output the revised Python code now:
```python
[Your revised code here]
```"""