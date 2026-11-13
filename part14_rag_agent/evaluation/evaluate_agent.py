"""Small manual agent audit; executes curated cases against local Ollama/DB."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage, ToolMessage
from graph import graph


def main():
    cases = json.loads((Path(__file__).parent / "test_cases.json").read_text())
    for case in cases:
        if case.get("inject_failure"):
            print(f"{case['id']}: skipped (failure injection is covered by unit tests)")
            continue
        result = graph.invoke({"messages": [HumanMessage(content=case["question"])]},
                              config={"configurable": {"thread_id": f"eval-{case['id']}"}})
        observed = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
        missing = sorted(set(case["expected_tools"]) - set(observed))
        print(f"{case['id']}: tools={observed}; missing_expected={missing}; answer={result['messages'][-1].content}")


if __name__ == "__main__":
    main()
