import json
import sys
import unittest
from pathlib import Path
from langchain_core.messages import ToolMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from graph import validate_citations


class GroundingTests(unittest.TestCase):
    def test_citations_must_exist_in_real_tool_output(self):
        tool = ToolMessage(name="search_knowledge_base", tool_call_id="1", content=json.dumps({
            "status": "relevant", "sources": [{"source_id": "S1", "source": "policy.txt", "chunk_id": 7}]
        }))
        answer, citations = validate_citations([tool], "20 days [S1]. Unknown [S9].")
        self.assertEqual(answer, "20 days [S1]. Unknown .")
        self.assertEqual([s["chunk_id"] for s in citations], [7])

    def test_error_tool_output_cannot_supply_citation(self):
        tool = ToolMessage(name="search_knowledge_base", tool_call_id="1", content='{"status":"error","sources":[]}')
        _, citations = validate_citations([tool], "Claim [S1]")
        self.assertEqual(citations, [])


if __name__ == "__main__":
    unittest.main()
