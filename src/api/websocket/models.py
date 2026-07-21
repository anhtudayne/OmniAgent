"""Định nghĩa các message types và data models cho WebSocket protocol."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from fastapi import WebSocket


class MessageType(str, Enum):
    """Enum các loại message trong WebSocket protocol.

    Chia làm 2 nhóm:
    - Client -> Server: user_message, tool_response
    - Server -> Client: connected, status, tool_request, final_response, error
    """

    # Client -> Server
    USER_MESSAGE = "user_message"
    TOOL_RESPONSE = "tool_response"

    # Server -> Client
    CONNECTED = "connected"
    STATUS = "status"
    TOOL_REQUEST = "tool_request"
    FINAL_RESPONSE = "final_response"
    ERROR = "error"


@dataclass
class ConnectionInfo:
    """Thông tin của 1 WebSocket connection.

    Attributes:
        connection_id: UUID định danh duy nhất cho connection.
        websocket: WebSocket object để gửi/nhận dữ liệu.
        created_at: Thời điểm tạo connection.
        is_busy: True khi đang xử lý prompt -> lock, không nhận prompt mới.
    """

    connection_id: str
    websocket: WebSocket
    created_at: datetime = field(default_factory=datetime.now)
    is_busy: bool = False
