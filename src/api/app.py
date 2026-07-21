"""FastAPI app with WebSocket endpoint for AI Assistant Server."""

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.api.websocket.connection_manager import ConnectionManager
from src.api.websocket.handler import handle_user_message, clear_conversation_history
from src.api.websocket.models import MessageType
from src.api.websocket.pending_manager import pending_manager

app = FastAPI(
    title="AI Assistant Server",
    description="Backend AI Assistant API & WebSocket Server",
    version="0.1.0",
)

manager = ConnectionManager()


@app.get("/health")
async def health_check():
    """Endpoint to check the server health status."""
    return {
        "status": "ok",
        "message": "AI Assistant Server is running",
        "active_connections": manager.active_count,
    }


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to handle connection lifecycle and client messages.

    Flow:
    1. Client connects -> Server assigns UUID and sends welcome message.
    2. Client sends user_message -> Server checks busy lock.
    3. If not busy -> Spawn async task to invoke LangGraph agent.
    4. If busy -> Return error requesting client to wait.
    """
    connection_id = await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == MessageType.USER_MESSAGE:
                content = data.get("content", "").strip()

                if not content:
                    await manager.send_to(connection_id, {
                        "type": MessageType.ERROR,
                        "content": "Message content cannot be empty.",
                    })
                    continue

                if manager.is_busy(connection_id):
                    await manager.send_to(connection_id, {
                        "type": MessageType.ERROR,
                        "content": "Please wait for the current request to complete.",
                    })
                    continue

                asyncio.create_task(
                    handle_user_message(manager, connection_id, content)
                )

            elif msg_type == MessageType.TOOL_RESPONSE:
                request_id = data.get("request_id", "")
                result = data.get("result")

                if not request_id:
                    await manager.send_to(connection_id, {
                        "type": MessageType.ERROR,
                        "content": "Missing 'request_id' in tool_response.",
                    })
                    continue

                pending_manager.resolve(request_id, result)

            else:
                await manager.send_to(connection_id, {
                    "type": MessageType.ERROR,
                    "content": f"Unknown message type: '{msg_type}'",
                })

    except WebSocketDisconnect:
        manager.disconnect(connection_id)
        pending_manager.cancel_all(f"Client {connection_id} disconnected")
        clear_conversation_history(connection_id)
    except Exception as e:
        manager.disconnect(connection_id)
        pending_manager.cancel_all(f"Client {connection_id} disconnected due to error: {e}")
        clear_conversation_history(connection_id)
        print(f"[WS] Error on connection {connection_id}: {e}")

