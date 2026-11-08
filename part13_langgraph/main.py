from langchain_core.messages import HumanMessage
from graph import graph
import uuid


def main():
    # Each application session gets a unique thread ID
    thread_id = "pavan-session-1"
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    print("\nLangGraph Agent Started!")
    print("Thread ID:", thread_id)
    print("Type 'exit' to quit or 'clear' to reset conversation.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Agent stopped.")
            break

        if user_input.lower() == "clear":
            # Start a fresh conversation
            thread_id = str(uuid.uuid4())

            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }

            print("\nConversation cleared.")
            print("New Thread ID:", thread_id, "\n")
            continue

        if not user_input:
            continue

        try:
            result = graph.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config
            )

            print("\nAgent:", result["messages"][-1].content, "\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()