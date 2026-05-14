from utils.ollama_client import query_ollama
from rag.vector_store import retrieve


def generate_final_report(
    query,
    extracted_data,
    critique,
    previous_report="",
    search_results=None       
):
    print(f"Previous report length: {len(previous_report)}")

    prior_context = ""
    if previous_report:
        prior_context = f"""
PREVIOUS REPORT (user is following up on this):
{previous_report}

Build on it, do not repeat it.
"""

    retrieved = retrieve(query)
    context = retrieved["documents"][0]

    sources_block = "\n".join(
        f"- {r['title']}: {r['link']}"
        for r in search_results
        if r.get("link")
    ) if search_results else "No sources available"

    prompt = f"""
    Generate a professional research report.

    {prior_context}

    USER QUERY:
    {query}

    RETRIEVED CONTEXT:
    {context}

    EXTRACTED FINDINGS:
    {extracted_data}

    CRITIQUE:
    {critique}

    SOURCES (use exact titles and links, do not invent dates):
    {sources_block}

    Structure:
    1. Executive Summary
    2. Core Strategy
    3. Technical Analysis
    4. Competitive Landscape
    5. Risks and Challenges
    6. Future Outlook

    Use concise analytical writing.
    Avoid generic statements.
    Do not write a Conclusion section.
    Do not use the phrases "in conclusion", "to summarize", or "overall".
    When listing references, use the source title and URL above.
    Do not add publication dates unless explicitly stated in the content.
    """

    return query_ollama(prompt, model="llama3")
