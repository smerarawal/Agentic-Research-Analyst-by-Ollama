from crewai import Task

NO_CHAIN_OF_THOUGHT = (
    "IMPORTANT: Output only the final answer. "
    "Do not include 'Thought:', 'Final Answer:', or any reasoning steps in your response."
)


def plan_task(agent, query):
    return Task(
        description=(
            f"Create 4 specific, web-searchable research tasks for this query: {query}\n"
            "Rules:\n"
            "- Use short keyword-based phrases, not full sentences\n"
            "- Include company/product/entity names\n"
            "- Avoid vague words like 'analyze', 'research', 'study'\n"
            "- Focus on technical, financial, competitive, or strategic angles\n\n"
            f"{NO_CHAIN_OF_THOUGHT}"
        ),
        expected_output=(
            "A numbered list of 4 specific search tasks, "
            "each one a short keyword phrase ready to paste into a search engine."
        ),
        agent=agent
    )


def research_task(agent, query, raw_content, context_tasks):
    return Task(
        description=(
            f"Extract concrete facts for: {query}\n\n"
            f"SOURCE MATERIAL:\n{raw_content}\n\n"
            "For each finding include:\n"
            "- The specific claim or data point\n"
            "- The source title it comes from\n"
            "- Why it is relevant to the query\n\n"
            "Rules:\n"
            "- Do NOT summarise. Do NOT say 'the article discusses'.\n"
            "- Pull out named figures, dates, product names, deal values.\n"
            "- If a fact is not in the source material, omit it. "
            "Never infer or assume technical details.\n\n"
            f"{NO_CHAIN_OF_THOUGHT}"
        ),
        expected_output=(
            "Bullet points of specific, verifiable facts with source titles. "
            "Each bullet names a concrete data point, not a general theme."
        ),
        agent=agent,
        context=context_tasks
    )


def critique_task(agent, context_tasks):
    return Task(
        description=(
            "Critique the research findings above.\n"
            "Identify:\n"
            "- At least 2 specific missing facts or data points\n"
            "- Any unsupported claims\n"
            "- Any important angles not covered\n\n"
            "Rules:\n"
            "- Be specific — name exactly what is missing, not just that something is missing\n"
            "- If a claim is not backed by a named source in the findings, flag it as unsupported\n\n"
            f"{NO_CHAIN_OF_THOUGHT}"
        ),
        expected_output=(
            "A structured critique with:\n"
            "1. Missing information (specific named gaps)\n"
            "2. Weak or unsupported claims\n"
            "3. Suggestions for what additional research would strengthen the report"
        ),
        agent=agent,
        context=context_tasks
    )


def report_task(agent, query, context_tasks, prior_context=""):
    return Task(
        description=(
            f"{prior_context}\n"
            f"Write a professional research report for: {query}\n\n"
            "Structure (use these exact headings):\n"
            "1. Executive Summary\n"
            "2. Core Strategy\n"
            "3. Technical Analysis\n"
            "4. Competitive Landscape\n"
            "5. Risks and Challenges\n"
            "6. Future Outlook\n\n"
            "Rules:\n"
            "- Use only facts that appear in the research findings\n"
            "- If a fact is not in the source material, omit it entirely\n"
            "- Never infer or assume technical details not explicitly stated\n"
            "- Do not write a Conclusion section\n"
            "- Do not use 'in conclusion', 'to summarize', or 'overall'\n"
            "- End after Future Outlook with a specific forward-looking statement\n"
            "- References must use this exact format: Source Title — URL\n"
            "- Do NOT add any dates, parentheses, or 'Retrieved from' to references\n"
            "- If you do not have a URL for a source, omit it entirely\n\n"
            f"{NO_CHAIN_OF_THOUGHT}"
        ),
        expected_output=(
            "A full structured research report in plain text with all 6 sections, "
            "backed by specific facts and named sources. "
            "References listed as 'Source Title — URL' with no dates."
        ),
        agent=agent,
        context=context_tasks
    )