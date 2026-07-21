"""Define local tools for the ReAct agent."""
import asyncio
import logging

from src.agent.tools.remote_tool import execute_remote_tool_async, ToolTimeoutError
from src.agent.tools.web_search.client import search_datalogic_knowledge
from src.agent.tools.web_search.schema import WebSource

logger = logging.getLogger(__name__)

_rag_instance = None
def get_qa_retriever(query: str) -> str:
    global _rag_instance

    logger.info(f'get_qa_retriever called with query: {query}')
    return "[RAG ERROR] RAG retriever is currently disabled. Please enable it to use this feature."
    # try:
    #     if _rag_instance is None:
    #         _rag_instance = RAGPipeline()
    #         _rag_instance.initialize()

    #     answer = _rag_instance.query(query)
    #     return answer
    # except Exception as e:
    #     logger.error(f'RAG retriever error: {e}')
    #     return f"[RAG ERROR] {str(e)}"

_device_session_history: dict[str, list] = {}
def get_sub_agent(query: str, connection_id: str = None, loop=None, conversation_history: list = None) -> str:
    from src.agent.tools.sub_agent.action_agent import device_control_node

    global _device_session_history

    conn_key = connection_id or "default"
    history = _device_session_history.get(conn_key, [])
    # logger.info(f"history hiện tại khi ở get_sub_agent là : {history}")
    state = {
        "query": query,
        "device_agent_history": history,
        "conversation_history": conversation_history or [],
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    out = device_control_node(state)
    _device_session_history[conn_key] = out.get("device_agent_history", [])

    result = out.get("device_agent_result") or {}
    result_type = result.get("type")
    print(f"[root-agent] result: {result}")
    if result_type == "action":
        action_name = result.get("action", "")

        if connection_id and loop:
            coro = execute_remote_tool_async(connection_id, result)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                ws_result = future.result()
                logger.info(f"[get_sub_agent] Angular response: {ws_result}")
            except ToolTimeoutError:
                raise
            except Exception as e:
                logger.error(f"[get_sub_agent] WebSocket send error: {e}")
        else:
            logger.warning("[get_sub_agent] No connection_id/loop — cannot send to Angular.")

        ws_result_str = ""
        if 'ws_result' in locals() and ws_result:
            ws_result_str = f" Result from client: {ws_result}"

        if action_name == "CONNECT_DEVICE":
            device_name = result.get("parameters", {}).get("deviceName", "")
            return (
                f"ACTION_DONE: Đã gửi lệnh kết nối thiết bị"
                + (f" '{device_name}'" if device_name else "")
                + f" xuống ứng dụng.{ws_result_str}"
            )
        elif action_name == "WRITE_CONFIG":
            return f"ACTION_DONE: Đã gửi lệnh ghi cấu hình xuống thiết bị.{ws_result_str}"
        else:
            return (
                f"ACTION_DONE: Đã thực thi lệnh '{result.get('name_function')}' "
                f"code={result.get('code')}, "
                f"params={result.get('parameters', {})}.{ws_result_str}"
            )
    elif result_type == "clarify":
        return f"   : {result.get('question', '')}"
    elif result_type == "info":
        return f"INFO: {result.get('answer', '')}"
    elif result_type == "error":
        return f"ERROR: {result.get('message', 'Lỗi không xác định.')}"
    else:
        return f"ERROR: {result.get('message', 'Lỗi không xác định.')}"

def web_retrieval(query: str) -> list[WebSource]:
    """Answer a Datalogic-related question using web search."""
    logger.info(f"web_retrieval called with question: {query}")
    context = search_datalogic_knowledge(query)
    print(context)
    return context

# tools = [get_system_info, get_current_config]

AGENT_TOOLS_LIST = {
    "TOOLS": [
        {
            "name": "get_qa_retriever",
            "description": (
                "Tra cứu thông tin/FAQ về cách sử dụng ứng dụng, tính năng, "
                "hướng dẫn... Dùng khi người dùng HỎI THÔNG TIN, không phải "
                "yêu cầu thực thi hành động."
            ),
            "args": {"query": "câu hỏi cần tra cứu"}
        },
        {
            "name": "get_sub_agent",
            "description": (
                "Chuyển câu lệnh cho sub-agent xử lý khi người dùng muốn "
                "THỰC THI/ĐIỀU KHIỂN một chức năng của ứng dụng hoặc thiết bị "
                "(vd: tăng âm lượng, bật wifi, đổi theme, mở app, hẹn giờ...)."
            ),
            "args": {"query": "câu lệnh gốc của người dùng, giữ nguyên văn"}
        },
        {
            "name": "web_retrieval",
            "description": (
                """
                Dùng để thu thập thông tin từ các trang web v Datalogic để tr lời câu hỏi của người dùng.
                CHỈ NÊN SỬ DỤNG tool này khi hệ thống RAG KHÔNG THỂ LẤY thông tin giúp ích cho việc trả lời câu hỏi của người dùng
                """
            ),
            "args": {"query": "câu hỏi cần tra cứu"}
        },
    ]
}

TOOLS_MAPPING_TO_FUNC = {
    "get_qa_retriever": get_qa_retriever,
    "get_sub_agent": get_sub_agent,
    "web_retrieval": web_retrieval,
}