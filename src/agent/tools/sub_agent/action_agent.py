import json
import logging
import re

from langgraph.graph import END, StateGraph

from src.agent.prompts.Instructions import SUB_AGENT_INSTRUCTION as AGENT_INSTRUCTION
from src.agent.tools.sub_agent.llm import Config, Gemini
from src.agent.tools.sub_agent.sub_tool import get_action_context, get_qa_retriever, AGENT_TOOLS_LIST
from src.agent.tools.sub_agent.state import AgentState

config = Config()
gemini_model = Gemini(config)
logger = logging.getLogger(__name__)


def format_tools_list() -> str:
    tools_str = ""
    for tool in AGENT_TOOLS_LIST["TOOLS"]:
        tools_str += f"- Tên tool: {tool['name']}\n"
        tools_str += f"  Mô tả: {tool['description']}\n"
        tools_str += f"  Tham số (args): {json.dumps(tool['args'], ensure_ascii=False)}\n\n"
    return tools_str


def call_agent(state: AgentState) -> AgentState:
    if not state.get('num_steps', 0):
        state.setdefault('history', []).append(
            {"role": "user", "content": state.get('query', '')}
        )
        
    logger.info(f"history hiện tại khi ở sub-agent là : {state.get('history')}")
    
    history_text = '\n'.join(
        f"{h['role'].upper()}: {h['content']}" for h in state['history'][:-1]
    ) or 'None yet - first turn'
    
    conv_history = state.get('conversation_history', [])
    if conv_history:
        conv_lines = []
        for turn in conv_history:
            conv_lines.append(f"User: {turn['user']}")
            conv_lines.append(f"Bot: {turn['bot']}")
            conv_lines.append("---")
        conv_text = '\n'.join(conv_lines)
    else:
        conv_text = 'Không có.'

    tools_list = format_tools_list()
    last_msg = state['history'][-1]['content']

    prompt = f"""
{AGENT_INSTRUCTION}

DANH SÁCH TOOLS HỖ TRỢ:
{tools_list}

RECENT CONVERSATION (lịch sử chat gần nhất với user):
{conv_text}

CLARIFICATION HISTORY (các bước đã chạy và context đã lấy):
{history_text}

TÌNH TRẠNG / CÂU HỎI HIỆN TẠI: {last_msg}

Trả lời:
"""
    response = gemini_model.invoke(prompt=prompt)
    state['last_response'] = response
    state['num_steps'] = state.get('num_steps', 0) + 1

    print(f'\n=== DEVICE-AGENT STEP {state["num_steps"]} ===')
    print(response)
    print('=' * 50)

    return state


def parse_response(state: AgentState) -> AgentState:
    raw = state.get('last_response', '')
    if not isinstance(raw, str):
        if isinstance(raw, dict):
            raw = raw.get('content', '')
        else:
            raw = getattr(raw, 'content', '')
            
    raw = raw.strip()
    
    action_name = None
    args = {}
    args_str = "{}"
    
    # Dọn dẹp sơ bộ nếu AI lỡ bọc code trong Markdown (```json)
    raw = re.sub(r'```json|```', '', raw).strip()
    
    # Khởi tạo các biến mặc định
    action_name = None
    args = {}
    args_str = "{}"
    
    # Dùng Regex để quét dữ liệu (Biểu thức này chấp nhận cả định dạng JSON lẫn Text thường)
    action_match = re.search(r'\"?ACTION\"?\s*:\s*\"?([^",\n\r]+)', raw, re.IGNORECASE)
    args_match = re.search(r'\"?ARGUMENTS\"?\s*:\s*(\{.*?\})', raw, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'\"?ANSWER\"?\s*:\s*\"?(.*?)\"?\s*\}?$', raw, re.DOTALL | re.IGNORECASE)
    clarify_match = re.search(r'\"?CLARIFY\"?\s*:\s*\"?(.*?)\"?\s*\}?$', raw, re.DOTALL | re.IGNORECASE)

    # Phân loại và lấy dữ liệu
    if action_match and args_match:
        action_name = action_match.group(1).strip()
        args_str = args_match.group(1).strip()
        try:
            # Ép kiểu chuỗi argument thành Dictionary
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
            
    elif answer_match:
        answer_text = answer_match.group(1).strip('", \n\r')
        
    elif clarify_match:
        clarify_text = clarify_match.group(1).strip('", \n\r')

    if action_name:
        state['parsed_action'] = action_name
        state['parsed_args'] = args
        
        if action_name in ["get_qa_retriever", "get_action_context"]:
            state['action_type'] = "internal"
            state.setdefault('history', []).append(
                {"role": "assistant", "content": f"ACTION: {action_name}\nARGUMENTS: {args_str}"}
            )
        elif action_name in ["detect_device", "connect_device", "write_config"]:
            state['action_type'] = "external"
            from src.agent.tools.sub_agent.sub_tool import TOOLS_MAPPING_TO_FUNC
            
            func = TOOLS_MAPPING_TO_FUNC.get(action_name)
            if func:
                data = func(**args)
            else:
                # Fallback if function missing
                data = {"type": "action", "name_function": action_name, "parameters": args}
            
            state['result'] = data
            state.setdefault('history', []).append(
                {"role": "assistant", "content": f"ACTION_OUTPUT: {json.dumps(data, ensure_ascii=False)}"}
            )
        else:
            state['action_type'] = "external"
            code = args.pop("code", "")
            data = {
                "type": "action", 
                "name_function": action_name, 
                "code": code,
                "parameters": args
            }
                
            state['result'] = data
            state.setdefault('history', []).append(
                {"role": "assistant", "content": f"ACTION_OUTPUT: {json.dumps(data, ensure_ascii=False)}"}
            )
            
    elif answer_match:
        data = {"type": "info", "answer": answer_match.group(1).strip()}
        state['result'] = data
        state['action_type'] = "end"
        
    elif clarify_match:
        data = {"type": "clarify", "question": clarify_match.group(1).strip()}
        state['result'] = data
        state['action_type'] = "clarify"
        state.setdefault('history', []).append(
            {"role": "assistant", "content": data['question']}
        )
    else:
        state['result'] = {
            "type": "error",
            "message": "Phản hồi không đúng định dạng."
        }
        state['action_type'] = "error"
    print(f"Kết quả của sub-agent là: {state['result']}")
    return state


def execute_internal_tool(state: AgentState) -> AgentState:
    action_name = state.get('parsed_action')
    args = state.get('parsed_args', {})
    
    if action_name == "get_qa_retriever":
        res = get_qa_retriever(args.get('query', ''))
    elif action_name == "get_action_context":
        res = get_action_context(args.get('query', ''))
    else:
        res = "Tool không hợp lệ."
        
    state.setdefault('history', []).append(
        {"role": "user", "content": f"TOOL OBSERVATION: {res}"}
    )
    return state


def should_continue(state: AgentState) -> str:
    action_type = state.get('action_type')
    if action_type == 'internal':
        return 'execute_internal_tool'
    return 'end'


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node('agent', call_agent)
    workflow.add_node('parse', parse_response)
    workflow.add_node('execute_internal_tool', execute_internal_tool)

    workflow.set_entry_point('agent')
    workflow.add_edge('agent', 'parse')

    workflow.add_conditional_edges(
        'parse',
        should_continue,
        {
            'execute_internal_tool': 'execute_internal_tool',
            'end': END,
        }
    )
    workflow.add_edge('execute_internal_tool', 'agent')

    return workflow.compile()


_device_control_graph = build_graph()


def device_control_node(state: dict) -> dict:
    sub_state = {
        "query": state.get("query", ""),
        "history": state.get("device_agent_history", []),
        "conversation_history": state.get("conversation_history", []),
        "last_response": "",
        "result": None,
        "num_steps": 0,
    }

    final_sub_state = _device_control_graph.invoke(sub_state)

    return {
        "device_agent_result": final_sub_state.get("result", {}),
        "device_agent_history": final_sub_state.get("history", []),
    }

def save_command_json(command: dict, path: str = 'command.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(command, f, ensure_ascii=False, indent=2)
    logger.info(f'Đã lưu lệnh vào {path}')
