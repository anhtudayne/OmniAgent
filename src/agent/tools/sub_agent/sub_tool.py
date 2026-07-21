"""Define local tools for the ReAct agent."""
import logging
import os
import sys
from src.agent.tools.sub_agent.RAG import RAGPipeline
logger = logging.getLogger(__name__)

_rag_instance = None

def _get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
        _rag_instance.initialize()
    return _rag_instance


def get_qa_retriever(query: str) -> str:
    logger.info(f'get_qa_retriever called with query: {query}')
    try:
        return _get_rag().get_action_context(query)
    except Exception as e:
        logger.error(f'get_action_context error: {e}')
        return f"[RAG ERROR] {str(e)}"


def get_action_context(query: str) -> str:
    print("Đang gọi get_action_context với query")
    logger.info(f'get_action_context called with query: {query}')
    try:
        return _get_rag().get_action_context(query)
    except Exception as e:
        logger.error(f'get_action_context error: {e}')
        return f"[RAG ERROR] {str(e)}"
def detect_device():
    return {
        "type": "action",
        "name_function": "detect_device",
        "parameters": {}
    }

def connect_device(deviceName: str = ""):
    return {
        "type": "action",
        "action": "CONNECT_DEVICE",
        "parameters": {
            "deviceName": deviceName
        }
    }

def write_config():
    return {
        "type": "action",
        "action": "WRITE_CONFIG",
        "parameters": {}
    }


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
            "name": "get_action_context",
            "description": (
                "Tra cứu tài liệu FRS (Functional Requirements Specification) "
                "để tìm mã (hex code) và các giá trị (options) của một tham số "
                "cấu hình máy quét Datalogic Magellan cụ thể."
            ),
            "args": {"query": "truy vấn chứa tên tham số hoặc chức năng cần cấu hình"}
        },
        {
            "name": "detect_device",
            "description": (
                "Quét và phát hiện thiết bị máy quét đang được kết nối với hệ thống. "
                "CHỈ GỌI hàm này khi người dùng yêu cầu thực thi/cấu hình mà hệ thống CHƯA detect thiết bị."
            ),
            "args": {}
        },
        {
            "name": "connect_device",
            "description": (
                "Kết nối với thiết bị máy quét đã được phát hiện. "
                "CHỈ GỌI hàm này sau khi detect_device thành công nhưng chưa connect."
            ),
            "args": {"deviceName": "Tên thiết bị lấy từ thông báo detect thành công"}
        },
        {
            "name": "write_config",
            "description": (
                "Lưu hoặc ghi toàn bộ cấu hình đã thay đổi xuống thiết bị. "
                "CHỈ GỌI khi người dùng yêu cầu lưu cấu hình."
            ),
            "args": {}
        }
    ]
}

TOOLS_MAPPING_TO_FUNC = {
    "get_qa_retriever": get_qa_retriever,
    "get_action_context": get_action_context,
    "detect_device": detect_device,
    "connect_device": connect_device,
    "write_config": write_config
}