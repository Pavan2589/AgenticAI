"""Live grounded-answer audit; compare output to curated expected answers."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage
from graph import graph, validate_citations


def main():
    cases = json.loads((Path(__file__).parent / "test_cases.json").read_text())
    selected = [case for case in cases if case["category"] == "grounding"]
    for case in selected:
        result = graph.invoke({"messages": [HumanMessage(content=case["question"])]},
                              config={"configurable": {"thread_id": f"grounding-{case['id']}"}})
        answer, citations = validate_citations(result["messages"], result["messages"][-1].content)
        print(f"\n{case['id']}\nExpected: {case['expected_answer']}\nObserved: {answer}\nValidated citations: {citations}")
        print("Review answer support and completeness manually.")


if __name__ == "__main__":
    main()
