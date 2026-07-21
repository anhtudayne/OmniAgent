"""Manage pending tool requests waiting for Angular client responses."""

import asyncio
from typing import Any, Dict, Tuple
from uuid import uuid4


class PendingRequestManager:
    """Manage asyncio.Future objects to pause/resume Agent."""

    def __init__(self):
        self._pending: Dict[str, asyncio.Future] = {}

    def create_request(self) -> Tuple[str, asyncio.Future]:
        """Create a new request_id and asyncio.Future."""
        request_id = str(uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        return request_id, future
 
    def resolve(self, request_id: str, result: Any):
        """Resolve the future when receiving response from client."""
        future = self._pending.pop(request_id, None)
        if future and not future.done():
            future.set_result(result)

    def cancel_request(self, request_id: str):
        """Remove a single timed-out request from the pending dict."""
        self._pending.pop(request_id, None)

    def cancel_all(self, error_msg: str = "Connection lost"):
        """Cancel all pending requests when connection is disconnected."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(Exception(error_msg))
        self._pending.clear()


pending_manager = PendingRequestManager()
