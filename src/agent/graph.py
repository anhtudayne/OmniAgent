"""Define LangGraph ReAct agent workflow."""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.nodes.node import call_agent, call_tool

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node('agent', call_agent)
    workflow.add_node('tools', call_tool)

    workflow.set_entry_point('agent')

    workflow.add_conditional_edges(
        'agent',
        should_continue,
        {
            'continue': 'tools',
            'end': END
        }
    )

    workflow.add_edge('tools', 'agent')

    return workflow.compile()


def should_continue(state: AgentState) -> str:
    response = state.get("last_agent_response", "").upper()

    if "ANSWER:" in response:
        print("Routing to END (found ANSWER)")
        return "end"

    if "ACTION:" in response:
        if state.get("num_steps", 0) >= 5:
            print("→ Routing to END (max steps reached)")
            return "end"
        print("Routing to TOOLS (found ACTION)")
        return "continue"

    print("Routing to END (no action found)")
    return "end"
