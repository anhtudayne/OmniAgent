"""AI Assistant - ReAct Agent Demo

Entry point để chạy agent tương tác trên terminal.
"""

import sys
import os

# Thêm thư mục gốc project vào sys.path để import 'from src.agent...' hoạt động
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config.settings import settings
from langchain_core.messages import HumanMessage
from src.agent.graph import graph


def main():
    """Chạy ReAct agent trong vòng lặp tương tác trên terminal hoặc khởi động server."""
    # Nếu truyền tham số --server, khởi động uvicorn server
    if "--server" in sys.argv:
        import uvicorn
        print(f"[Server] Starting AI Assistant Server on http://{settings.API_HOST}:{settings.API_PORT}...")
        uvicorn.run("src.api.app:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
        return

    # Kiểm tra API key (chỉ bắt buộc trong chế độ CLI/Agent)
    if not settings.LLM_API_KEY or settings.LLM_API_KEY == "gsk_your-groq-api-key-here":
        print("❌ Error: Please update LLM_API_KEY in the .env file")
        sys.exit(1)

    print()
    print("=" * 55)
    print("  🤖  AI Assistant - ReAct Agent Demo")
    print("=" * 55)
    print()
    print("  Welcome! I am the company's AI Assistant.")
    print("  I can answer questions and perform tasks for you.")
    print()
    print("  💡 Try asking:")
    print('     • "Show me the system information"')
    print('     • "What is the current website configuration?"')
    print('     • "Hello"')
    print()
    print("  Type 'quit' or 'exit' to quit.")
    print("=" * 55)
    print()

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye! See you later.\n")
                break

            print("\n⏳ Processing...\n")

            # Gọi ReAct agent graph (recursion_limit giới hạn số vòng lặp ReAct)
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"recursion_limit": 10},
            )

            # Lấy response cuối cùng từ AI
            ai_message = result["messages"][-1]
            print(f"\n🤖 Assistant: {ai_message.content}\n")
            print("-" * 55)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! See you later.\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            print("-" * 55)
            print()


if __name__ == "__main__":
    main()
