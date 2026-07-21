from typing import TypedDict, Optional

class AgentState(TypedDict):
    query: str
    history: list
    conversation_history: list
    last_response: str
    result: Optional[dict]
    num_steps: int
    action_type: Optional[str]
    parsed_action: Optional[str]
    parsed_args: Optional[dict]
