"""Define remote tools to be executed on Angular client side."""

import asyncio
from src.api.websocket.pending_manager import pending_manager
from src.api.websocket.models import MessageType

TOOL_TIMEOUT_SECONDS = 30


class ToolTimeoutError(Exception):
    """Raised when a remote tool execution exceeds the configured timeout."""
    pass


async def execute_remote_tool_async(
    connection_id: str,
    command: dict,
) -> str:
    """Send a command (action-agent JSON output) via WebSocket and wait for client response."""
    from src.api.app import manager

    request_id, future = pending_manager.create_request()

    await manager.send_to(connection_id, {
        "type": MessageType.TOOL_REQUEST,
        "request_id": request_id,
        "command": command,
    })

    tool_name = command.get("name_function", "unknown")
    print(f"[Remote Tool] Sent request {request_id} for command '{tool_name}'")

    try:
        result = await asyncio.wait_for(future, timeout=TOOL_TIMEOUT_SECONDS)
        print(f"[Remote Tool] Received response for {request_id}: {result}")
        return str(result)
    except asyncio.TimeoutError:
        pending_manager.cancel_request(request_id)
        print(f"[Remote Tool] Timeout for request {request_id} after {TOOL_TIMEOUT_SECONDS}s")
        raise ToolTimeoutError(
            f"Command '{tool_name}' timed out after {TOOL_TIMEOUT_SECONDS} seconds. "
            f"The Angular client did not respond in time."
        )
    except Exception as e:
        print(f"[Remote Tool] Error executing command {request_id}: {e}")
        return f"Error: Command '{tool_name}' failed: {str(e)}"
