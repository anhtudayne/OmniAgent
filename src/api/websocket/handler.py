"""Business logic handler for WebSocket messages."""

import asyncio

from src.agent.graph import build_graph
from src.agent.tools.remote_tool import ToolTimeoutError
from src.api.websocket.connection_manager import ConnectionManager
from src.api.websocket.models import MessageType

_graph = build_graph()

HISTORY_MAX_TURNS = 3
_conversation_histories: dict[str, list] = {}


def clear_conversation_history(connection_id: str):
    _conversation_histories.pop(connection_id, None)


async def handle_user_message(
    manager: ConnectionManager,
    connection_id: str,
    content: str,
):
    """Invoke LangGraph Agent asynchronously to process prompt and return response."""
    manager.set_busy(connection_id, True)

    try:
        await manager.send_to(connection_id, {
            "type": MessageType.STATUS,
            "content": "Thinking...",
        })

        loop = asyncio.get_running_loop()

        history = _conversation_histories.get(connection_id, [])
 
        state = {
            "query": content,
            "last_agent_response": "",
            "tool_observations": [],
            "num_steps": 0,
            "conversation_history": history,
        }

        result = await asyncio.to_thread(
            _graph.invoke,
            state,
            config={
                "recursion_limit": 10,
                "configurable": {
                    "connection_id": connection_id,
                    "loop": loop
                }
            },
        )

        raw_response = result.get("last_agent_response", "")
        if "ANSWER:" in raw_response:
            answer = raw_response.split("ANSWER:", 1)[1].strip()
        else:
            answer = raw_response

        history = history + [{"user": content, "bot": answer}]
        _conversation_histories[connection_id] = history[-HISTORY_MAX_TURNS:]

        await manager.send_to(connection_id, {
            "type": MessageType.FINAL_RESPONSE,
            "content": answer,
        })

    except ToolTimeoutError as e:
        print(f"[Handler] Tool timeout on connection {connection_id}: {e}")
        await manager.send_to(connection_id, {
            "type": MessageType.ERROR,
            "content": f"Timeout error: {str(e)}",
        })
    except Exception as e:
        print(f"[Handler] Error processing prompt for {connection_id}: {e}")
        await manager.send_to(connection_id, {
            "type": MessageType.ERROR,
            "content": f"An error occurred: {str(e)}",
        })

    finally:
        manager.set_busy(connection_id, False)
