from typing_extensions import TypedDict

class AgentState(TypedDict):
    query: str
    last_agent_response: str
    tool_observations: list
    num_steps: int
    conversation_history: list 
