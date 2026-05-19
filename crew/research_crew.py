from crewai import Crew, Process
from crew.agents import make_planner, make_researcher, make_critic, make_reporter
from crew.tasks import plan_task, research_task, critique_task, report_task
from agents.search_agent import search_web
from utils.web_scraper import scrape_webpage
from rag.vector_store import add_documents, get_relevant_context
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st


def clean_output(text: str) -> str:
    """Strip chain-of-thought leakage from local model outputs."""

    phrases_to_strip = [
        "I now can give a great answer",
        "Final Answer:",
        "Thought:",
    ]

    for phrase in phrases_to_strip:
        if text.strip().startswith(phrase):
            text = text[text.find(phrase) + len(phrase):].strip()

    return text.strip()


def fetch_content(query: str) -> str:
    """
    Search and scrape before the crew runs.
    Each page is capped at 1500 chars to avoid overflowing
    llama3's context window when passed as a prompt.
    """

    results = search_web(query)
    seen    = set()
    pages   = []

    def scrape(r):
        link = r["link"]
        if link in seen:
            return ""
        seen.add(link)
        text = scrape_webpage(link) or r.get("snippet", "")
        if not text:
            return ""
        # Cap each page at 1500 chars — RAG handles relevance,
        # no need to pass full pages to the LLM directly
        return f"Title: {r['title']}\nURL: {link}\n{text[:1500]}"

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(scrape, r) for r in results[:5]]
        for f in as_completed(futures):
            result = f.result()
            if result:
                pages.append(result)

    combined = "\n\n---\n\n".join(pages)
    print(f"\nFetched {len(pages)} pages, total chars: {len(combined)}\n")
    return combined


def run_research_crew(query: str, previous_report: str = "") -> str:

    prior_context = ""
    if previous_report:
        prior_context = (
            "PREVIOUS REPORT (user is following up on this):\n"
            f"{previous_report}\n\n"
            "Build on it, do not repeat it.\n"
        )

    # Step 1 — fetch real web content before crew starts
    raw_content = fetch_content(query)

    # Step 2 — embed scraped content into ChromaDB for RAG
    if raw_content.strip():
        add_documents([raw_content])
        print("Content embedded into ChromaDB.")

    # Step 3 — retrieve only the most relevant chunks
    # This is what gets passed to the researcher — not the full raw content.
    # Keeps the prompt small enough for llama3 to process without timing out.
    rag_context = get_relevant_context(query)
    print(f"RAG context retrieved: {len(rag_context)} chars")

    # Fall back to capped raw content if RAG returns nothing
    researcher_input = rag_context if rag_context.strip() else raw_content[:3000]

    planner    = make_planner()
    researcher = make_researcher()
    critic     = make_critic()
    reporter   = make_reporter()

    t_plan     = plan_task(planner, query)
    t_research = research_task(researcher, query, researcher_input, [t_plan])
    t_critique = critique_task(critic, [t_research])
    t_report   = report_task(reporter, query, [t_research, t_critique], prior_context)

    crew = Crew(
        agents=[planner, researcher, critic, reporter],
        tasks=[t_plan, t_research, t_critique, t_report],
        process=Process.sequential,
        verbose=True,
        memory=False
    )

    result = crew.kickoff(inputs={"query": query})

    # Strip chain-of-thought leakage then save to session state
    st.session_state["last_plan"] = (
        clean_output(t_plan.output.raw) if t_plan.output else "Not available"
    )
    st.session_state["last_extracted"] = (
        clean_output(t_research.output.raw) if t_research.output else "Not available"
    )
    st.session_state["last_critique"] = (
        clean_output(t_critique.output.raw) if t_critique.output else "Not available"
    )

    return clean_output(str(result))