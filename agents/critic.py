from utils.ollama_client import query_ollama


def critic_agent(report):

    prompt = f"""
    Critique this research report.

    Identify:
    - missing information
    - weak arguments
    - unsupported claims

    Report:
    {report}
    """

    return query_ollama(prompt, model="llama3")