"""LLM node and routing logic for ReAct agent."""
import logging
import json
from langchain_core.runnables import RunnableConfig
from src.agent.state import AgentState
from src.agent.tools.tool import TOOLS_MAPPING_TO_FUNC, AGENT_TOOLS_LIST
from src.agent.prompts.Instructions import ROOT_AGENT_INSTRUCTION as AGENT_INSTRUCTION
from src.config.llm import Config, Gemini

logger = logging.getLogger(__name__)
config = Config()
gemini_model = Gemini(config)
 
# all_tools = local_tools + remote_tools
# llm_with_tools = llm.bind_tools(all_tools)


# def llm_node(state: AgentState) -> dict:
#     """Invoke LLM with history and bound tools."""
#     messages = state["messages"]

#     if not messages or not isinstance(messages[0], SystemMessage):
#         messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

#     response = llm_with_tools.invoke(messages)
#     return {"messages": [response]}


# def should_continue(state: AgentState) -> str:
#     """Determine whether to continue execution with tools or terminate."""
#     last_message = state["messages"][-1]

#     if hasattr(last_message, "tool_calls") and last_message.tool_calls:
#         called_tools = set()
#         for msg in state["messages"]:
#             if isinstance(msg, ToolMessage):
#                 called_tools.add(msg.name)

#         requested_tools = {tc["name"] for tc in last_message.tool_calls}
#         if requested_tools.issubset(called_tools):
#             return "end"

#         return "tools"

#     return "end"

def build_tools_list() -> str:
    tools = AGENT_TOOLS_LIST.get('TOOLS', [])
    tool_line = ['Available tools:']
    for i, tool in enumerate(tools, 1):
        tool_line.append(
            f"[{i}]: {tool['name']}\n"
            f"    Description: {tool['description']}\n"
            f"    Arguments: {tool['args']}"
        )
    return '\n'.join(tool_line)

def call_agent(state: AgentState) -> AgentState:
    observations = '\n\n'.join(state.get('tool_observations', []))
    logger.info(f"[node.py-call_agent]observations: {observations}")
    if not observations:
        observations = 'None yet - first turn'


    tools_list = build_tools_list()

    # Count how many times each tool was called
    tool_calls_count = {}
    for obs in state.get('tool_observations', []):
        if 'TOOL:' in obs:
            tool_name = obs.split('TOOL:')[1].split('\n')[0].strip()
            tool_calls_count[tool_name] = tool_calls_count.get(tool_name, 0) + 1

    tools_used = ', '.join([f"{k}: {v}x" for k, v in tool_calls_count.items()]) if tool_calls_count else "None"

    # Format conversation history (3 lượt gần nhất)
    history = state.get('conversation_history', [])
    if history:
        history_lines = []
        for turn in history:
            history_lines.append(f"User: {turn['user']}")
            history_lines.append(f"Bot: {turn['bot']}")
            history_lines.append("---")
        history_text = '\n'.join(history_lines)
    else:
        history_text = 'Không có lịch sử trước đó.'
    logger.info(f"[node.py-call_agent]history_text: {history_text}")
    prompt = f"""
{AGENT_INSTRUCTION}

{tools_list}

CONVERSATION HISTORY (3 lượt gần nhất):
{history_text}

USER QUERY: {state.get('query')}

TOOLS ALREADY USED: {tools_used}

PAST TOOL OBSERVATIONS:
{observations}

IMPORTANT: If the observations above contain relevant information, ANSWER NOW. 
Do not call tools again unless absolutely necessary.

Respond now:
"""
    response = gemini_model.invoke(prompt=prompt)
    state['last_agent_response'] = response
    state['num_steps'] = state.get('num_steps', 0) + 1

    print(f'\n=== ROOT-AGENT STEP {state["num_steps"]} ===')
    print(response)
    print('=' * 50)

    return state


def call_tool(state: AgentState, config: RunnableConfig = None) -> AgentState:
    action_text = state.get('last_agent_response', '')
    logger.info(f"[node.py-call_tool]action_text: {action_text}")
    if 'ACTION:' not in action_text:
        state.setdefault('tool_observations', []).append('No ACTION found')
        return state

    # Lấy connection_id và loop từ LangGraph config
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else getattr(config, "get", lambda k, d=None: {})("configurable", {})
    connection_id = configurable.get("connection_id") if configurable else None
    loop = configurable.get("loop") if configurable else None

    try:
        # Extract tool name
        tool_name = None
        for line in action_text.split('\n'):
            if line.strip().startswith('ACTION:'):
                tool_name = line.split('ACTION:')[1].strip() # lấy cái tool ra, rag hoặc sub-agent
                break

        if not tool_name:
            state.setdefault('tool_observations', []).append('Could not extract tool name')
            return state

        # Extract arguments
        arguments = {}
        for line in action_text.split('\n'):
            if line.strip().startswith('ARGUMENTS:'):
                args_str = line.split('ARGUMENTS:')[1].strip()
                arguments = json.loads(args_str) # tách đối sổ ra, ở đây đối số  là query
                break

        # Get tool function
        tool_func = TOOLS_MAPPING_TO_FUNC.get(tool_name)

        if not tool_func:
            state.setdefault('tool_observations', []).append(f'Tool {tool_name} not found')
            return state

        # Execute tool — truyền connection_id và loop vào get_sub_agent
        print(f'\n>>> Executing tool: {tool_name} with args: {arguments}')
        if tool_name == 'get_sub_agent':
            result = tool_func(
                **arguments,
                connection_id=connection_id,
                loop=loop,
                conversation_history=state.get('conversation_history', []),
            )
        else:
            result = tool_func(**arguments)

        observation = f'TOOL: {tool_name}\nRESULT: {result}' # kết quả của tool, 1 là RAG 2 là sub-agent
        state.setdefault('tool_observations', []).append(observation) # đưa vào state quan sát

        logger.info(f'Tool {tool_name} executed successfully')

    except json.JSONDecodeError as e:
        state.setdefault('tool_observations', []).append(f'JSON parsing error: {str(e)}')
        logger.error(f'JSON error: {e}')
    except Exception as e:
        state.setdefault('tool_observations', []).append(f'Tool execution error: {str(e)}')
        logger.error(f'Tool execution error: {e}')

    return state


