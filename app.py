import streamlit as st

from agents.planner import planner_agent
from agents.search_agent import search_web
from agents.extractor import extract_information
from agents.critic import critic_agent
from agents.report_generator import generate_final_report
from agents.clarification_agent import needs_clarification

from memory import save_memory, get_memory


st.set_page_config(
    page_title="Agentic Research Analyst",
    layout="wide"
)

st.title("Agentic Research Analyst")

if "last_report" not in st.session_state:
    st.session_state.last_report = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

st.write(
    """
    Multi-agent research system using:
    - Local LLMs
    - RAG
    - Memory
    - Structured outputs
    """
)

# Shows whether previous context is loaded
if st.session_state.last_report:
    st.info(f"✓ Previous context loaded: \"{st.session_state.last_query}\"")
else:
    st.caption("No previous context — first query will start fresh.")

query = st.text_input(
    "Enter Research Query",
    placeholder="Analyze NVIDIA AI strategy in data center chips"
)

if st.button("Run Research"):

    if not query.strip():
        st.error("Please enter a query.")
        st.stop()

    if needs_clarification(query):
        st.warning(
            """
            Query is too broad.

            Please specify:
            - financial analysis
            - partnerships
            - AI products
            - acquisitions
            """
        )
        st.stop()

    with st.spinner("Planning tasks..."):
        try:
            plan = planner_agent(query)
        except Exception as e:
            st.error(f"Planner failed: {e}")
            st.stop()

    st.subheader("Research Plan")
    st.json(plan)

    # Search phase
    all_results = []

    with st.spinner("Searching sources..."):
        try:
            for task in plan["tasks"]:
                results = search_web(
                    f"{query} {task}"
                )
                all_results.extend(results)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    st.subheader("Retrieved Sources")

    with st.expander("View Retrieved Sources"):
        for idx, result in enumerate(all_results[:5]):
            st.write(f"Title: {result['title']}")
            st.write(f"Snippet: {result['snippet']}")
            st.write(f"Link: {result['link']}")
            st.write("---")

    # Extraction phase
    with st.spinner("Extracting insights..."):
        try:
            extracted = extract_information(
                all_results,
                query
            )
            save_memory(extracted)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

    st.subheader("Extracted Insights")
    st.write(extracted)

    # Critique phase
    with st.spinner("Critiquing findings..."):
        try:
            critique = critic_agent(extracted)
            save_memory(critique)
        except Exception as e:
            st.error(f"Critique failed: {e}")
            st.stop()

    st.subheader("Critique")

    if not critique:
        st.warning("Critic returned empty response")
    else:
        st.write(critique)

    # Report generation
    with st.spinner("Generating report..."):
        try:
            final_report = generate_final_report(
                query=query,
                extracted_data=extracted,
                critique=critique,
                previous_report=st.session_state.last_report,
                search_results=all_results
            )

            # Save to session state for follow-up queries
            st.session_state.last_report = final_report
            st.session_state.last_query = query

            save_memory(final_report)

        except Exception as e:
            st.error(f"Report generation failed: {e}")
            st.stop()

    st.subheader("Final Report")
    st.write(final_report)

    # Memory display
    st.subheader("Memory")
    memory_data = get_memory()
    st.write(memory_data)

    # Download option
    st.download_button(
        label="Download Report",
        data=final_report,
        file_name="research_report.txt",
        mime="text/plain"
    )
