from utils.ollama_client import query_ollama
from utils.json_parser import safe_json_parse


def planner_agent(query):

    prompt = f"""
    You are a research planning agent.

    Convert the user query into short web-searchable research tasks.

    Return ONLY valid JSON.

    Format:
    {{
        "tasks": [
            "task1",
            "task2"
        ]
    }}

    Rules:
    - Maximum 4 tasks
    - Use short keyword-based tasks
    - Include company/entity names
    - Avoid questions
    - Avoid generic words like:
      research
      study
      identify
      analyze
    - Focus on factual searchable topics
    - Focus on technical, business, financial, competitive, or strategic aspects

    User Query:
    {query}
    """

    response = query_ollama(prompt, model="mistral")

    return safe_json_parse(response)
