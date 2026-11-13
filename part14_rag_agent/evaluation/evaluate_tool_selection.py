"""Live evaluation of LangGraph tool selection against curated expectations."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage, ToolMessage
from graph import graph


def main():
    cases = json.loads((Path(__file__).parent / "test_cases.json").read_text())
    selected = [case for case in cases if case["category"] == "tool_selection"]
    passed = 0
    for case in selected:
        result = graph.invoke({"messages": [HumanMessage(content=case["question"])]},
                              config={"configurable": {"thread_id": f"tools-{case['id']}"}})
        observed = {m.name for m in result["messages"] if isinstance(m, ToolMessage)}
        expected = set(case["expected_tools"])
        ok = expected.issubset(observed)
        passed += ok
        print(f"{case['id']}: {'PASS' if ok else 'FAIL'} expected={sorted(expected)} observed={sorted(observed)}")
    print(f"Tool-selection cases passed: {passed}/{len(selected)}")


if __name__ == "__main__":
    main()
