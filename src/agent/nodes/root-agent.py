import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.graph import build_graph
from src.agent.tools.tool import get_sub_agent

load_dotenv()
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HISTORY_MAX_TURNS = 3
_cli_history: list = []

def run_query(query: str, graph) -> dict:
    global _cli_history

    state = {
        "query": query,
        "last_agent_response": "",
        "tool_observations": [],
        "num_steps": 0,
        "conversation_history": _cli_history,
    }

    result = graph.invoke(state)
    response = result.get('last_agent_response', '')

    if 'ANSWER:' in response:
        answer = response.split('ANSWER:', 1)[1].strip()
    else:
        answer = response

    _cli_history = (_cli_history + [{"user": query, "bot": answer}])[-HISTORY_MAX_TURNS:]

    pending_clarify = any(
        'NEED_CLARIFY:' in obs for obs in result.get('tool_observations', [])
    )

    return {"answer": answer, "pending_clarify": pending_clarify}


def main():
    print("Initializing App Assistant...")
    graph = build_graph()
    print("Ready! Type 'quit', 'exit', or 'esc' to stop.\n")

    # Cờ trạng thái session: True nghĩa là lượt trước sub-agent vừa hỏi lại
    # (clarify), nên lượt NÀY phải hiểu là câu trả lời cho câu hỏi đó.
    pending_subagent_clarify = False

    while True:
        query = input('User: ').strip()
        if query.lower() in ['quit', 'exit', 'esc']:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            if pending_subagent_clarify:
                # Bỏ qua hoàn toàn vòng suy luận ACTION của root — forward
                # thẳng câu trả lời của user cho sub-agent (nó tự nhớ history
                # riêng của nó qua _device_session_history trong Tools.py).
                observation = get_sub_agent(
                    query=query,
                    conversation_history=_cli_history
                )
                pending_subagent_clarify = observation.startswith('NEED_CLARIFY:')

                # Bóc phần message sau dấu ":" đầu tiên để in cho gọn.
                message = observation.split(':', 1)[1].strip() if ':' in observation else observation
                print(f'\nBot: {message}')
            else:
                result = run_query(query=query, graph=graph)
                pending_subagent_clarify = result['pending_clarify'] # được cập nhật nếu sub-agent vừa hỏi lại (clarify) trong lượt này, true hoặc false
                print(f'\nBot: {result["answer"]}')

            print('---' * 20 + '\n')
        except Exception as e:
            logger.error(f'Error processing query: {e}')
            print(f"Sorry, an error occurred: {e}\n")


if __name__ == '__main__':
    main()