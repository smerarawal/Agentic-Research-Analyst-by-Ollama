import streamlit as st

from workflow import MODEL, run_research


st.set_page_config(page_title="Local Research Analyst", layout="wide")
st.title("Local Research Analyst")
st.caption(f"Ollama model: `{MODEL}` · LangGraph workflow · evidence-checked citations")
st.write("Enter a focused research query. The planner creates sub-questions, the critic can request another search round, and code checks quotes and citations before showing the report.")

query = st.text_input("Research query", placeholder="NVIDIA Blackwell GPU deployments and announced customer systems, 2024–2025")

if st.button("Run research", type="primary", disabled=not query.strip()):
    with st.spinner("Researching with local Ollama models…"):
        try:
            result = run_research(query.strip())
        except Exception as exc:
            st.error(f"Research run failed: {exc}")
            st.stop()

    if not result.get("specific"):
        st.warning(result.get("clarification") or "Please make the query more specific.")
        st.stop()

    tabs = st.tabs(["Plan", "Sources", "Facts", "Critique", "Report", "Trace"])
    with tabs[0]:
        st.write("**Research query:**", result.get("query", query))
        st.write(result.get("sub_questions", []))
        st.caption(f"Critic search rounds: {max(0, result.get('iteration', 0) - 1)}")
    with tabs[1]:
        sources = result.get("sources", {})
        if not sources:
            st.warning("No pages or snippets were available for this query.")
        for source_id, source in sources.items():
            st.markdown(f"**[{source_id}] {source['title']}**  \n{source['url']}")
            st.caption(source.get("text", "")[:450].replace("\n", " ") + "…")
    with tabs[2]:
        facts = result.get("facts", [])
        if facts:
            for fact in facts:
                st.markdown(f"- **[{fact['source_id']}] {fact['claim']}**  \n  Quote: “{fact['quote']}”")
        else:
            st.info("No source quotes passed verification. The report should state that evidence is limited.")
    with tabs[3]:
        st.json(result.get("critique", {}))
    with tabs[4]:
        check = result.get("check", {})
        if check.get("passed"):
            st.success("Automated report checks passed.")
        else:
            st.warning("Automated checks found issues after one regeneration. Review the trace and sources before relying on this draft.")
            st.json(check)
        report = result.get("report", "")
        st.markdown(report)
        st.download_button("Download report", report, file_name="research_report.txt", mime="text/plain")
    with tabs[5]:
        st.caption(f"Run ID: {result.get('run_id')} · trace saved to `{result.get('trace_file')}`")
        st.json(result.get("trace", []))
        st.json({"model": MODEL, "citation_check": result.get("check", {})})
