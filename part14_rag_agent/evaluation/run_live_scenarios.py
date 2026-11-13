"""Run the requested seven end-to-end scenarios against local Ollama and PG."""
import sys
import uuid
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, ToolMessage
from graph import graph, validate_citations

SCENARIOS = [
    ("A RAG", "What is our company's leave policy?", {"search_knowledge_base"}),
    ("B calculator", "What is 125 multiplied by 48?", {"calculator"}),
    ("C multi-tool", "How many annual leaves remain under our current leave policy if an employee has used 6 days?", {"search_knowledge_base", "calculator"}),
    ("D missing", "What is our maternity leave policy?", {"search_knowledge_base"}),
    ("E unsupported", "Can unused leaves be converted into cash?", {"search_knowledge_base"}),
    ("F irrelevant", "What is the capital of Japan?", set()),
    ("G policy conflict", "What is the current annual leave allowance?", {"search_knowledge_base"}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["A", "B", "C", "D", "E", "F", "G"])
    parser.add_argument("--remaining", action="store_true", help="Run scenarios B through G")
    args = parser.parse_args()
    passed = 0
    cases = [case for case in SCENARIOS if not args.remaining or not case[0].startswith("A ")]
    if args.scenario:
        cases = [case for case in cases if case[0].startswith(args.scenario + " ")]
    for label, query, expected in cases:
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": f"part14-live-{uuid.uuid4()}"}},
        )
        messages = result["messages"]
        observed = {m.name for m in messages if isinstance(m, ToolMessage)}
        raw_answer = str(messages[-1].content)
        answer, citations = validate_citations(messages, raw_answer)
        ok = expected.issubset(observed) and observed.issubset(expected)
        passed += ok
        print(f"\n{label}: {'TOOL PASS' if ok else 'TOOL REVIEW'}")
        print(f"  expected tools: {sorted(expected)}")
        print(f"  observed tools: {sorted(observed)}")
        print(f"  answer: {answer}")
        print(f"  validated citations: {citations}", flush=True)
    print(f"\nTool-selection exact matches: {passed}/{len(cases)}")
    print("Review answer correctness and abstention manually against the curated expectations.")


if __name__ == "__main__":
    main()
