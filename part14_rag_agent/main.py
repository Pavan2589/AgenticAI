import uuid
from langchain_core.messages import HumanMessage
from graph import graph, validate_citations


def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print("Part 14 knowledge-grounded agent started.")
    print(f"Thread ID: {thread_id}. Type 'clear' for a new conversation or 'exit' to quit.")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() == "exit":
            break
        if prompt.lower() == "clear":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print(f"New conversation: {thread_id}")
            continue
        if not prompt:
            continue
        try:
            result = graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
            answer, citations = validate_citations(result["messages"], result["messages"][-1].content)
            print(f"\nAgent: {answer}")
            for source in citations:
                page = f", page {source['page_number']}" if source.get("page_number") else ""
                print(f"  [{source['source_id']}] {source['source']}{page} (chunk {source['chunk_id']}, similarity {source['similarity']:.3f})")
        except Exception as exc:
            print(f"\nAgent error: {exc}")


if __name__ == "__main__":
    main()
