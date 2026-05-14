from langgraph.graph import StateGraph


workflow = StateGraph(dict)


workflow.add_node("planner", planner_agent)
workflow.add_node("critic", critic_agent)

workflow.set_entry_point("planner")

workflow.add_edge(
    "planner",
    "critic"
)

graph = workflow.compile()