"""
Manage all active WebSocket connections.
"""

import uuid
from fastapi import WebSocket

from src.api.websocket.models import ConnectionInfo, MessageType


class ConnectionManager:
    """Manage WebSocket connections and busy status."""

    def __init__(self):
        self._connections: dict[str, ConnectionInfo] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Accept new connection, assign UUID, and send welcome message."""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self._connections[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            websocket=websocket,
        )

        await websocket.send_json({
            "type": MessageType.CONNECTED,
            "connection_id": connection_id,
        })

        print(f"[WS] Client connected: {connection_id}")
        return connection_id

    def disconnect(self, connection_id: str):
        """Remove connection info when client disconnects."""
        self._connections.pop(connection_id, None)
        print(f"[WS] Client disconnected: {connection_id}")

    async def send_to(self, connection_id: str, message: dict):
        """Send JSON message to a specific client."""
        conn = self._connections.get(connection_id)
        if conn:
            await conn.websocket.send_json(message)

    def get_connection(self, connection_id: str) -> ConnectionInfo | None:
        """Get connection info by ID."""
        return self._connections.get(connection_id)

    def set_busy(self, connection_id: str, busy: bool):
        """Set busy state to lock/unlock new prompt handling."""
        conn = self._connections.get(connection_id)
        if conn:
            conn.is_busy = busy

    def is_busy(self, connection_id: str) -> bool:
        """Check if the connection is currently busy processing AI request."""
        conn = self._connections.get(connection_id)
        return conn.is_busy if conn else False

    @property
    def active_count(self) -> int:
        """Return the count of active connections."""
        return len(self._connections)
