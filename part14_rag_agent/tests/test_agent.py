import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools as agent_tools
import graph as agent_graph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class AgentToolTests(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(agent_tools.safe_calculate("3 * 7"), 21)

    def test_calculator_rejects_code(self):
        with self.assertRaises(ValueError):
            agent_tools.safe_calculate("__import__('os').getcwd()")

    def test_knowledge_tool_is_registered(self):
        self.assertIn("search_knowledge_base", [tool.name for tool in agent_tools.tools])

    @patch.object(agent_graph, "_tool_selector")
    def test_json_planner_emits_langgraph_tool_calls(self, llm):
        llm.invoke.return_value = AIMessage(content='{"tool_calls":[{"name":"calculator","args":{"expression":"2+2"}}]}')
        result = agent_graph.chatbot({"messages": [HumanMessage(content="2+2?")]})
        self.assertEqual(result["messages"][0].content, "")
        self.assertEqual(result["messages"][0].tool_calls[0]["name"], "calculator")

    @patch.object(agent_graph, "_tool_selector")
    def test_planner_suppresses_already_executed_duplicate(self, llm):
        llm.invoke.return_value = AIMessage(content='{"tool_calls":[{"name":"calculator","args":{"expression":"20 - 6"}}]}')
        messages = [
            HumanMessage(content="How many days remain?"),
            AIMessage(content="", tool_calls=[{"name":"calculator","args":{"expression":"20 - 6"},"id":"call-1","type":"tool_call"}]),
            ToolMessage(name="calculator",tool_call_id="call-1",content="14"),
        ]
        result = agent_graph.chatbot({"messages": messages})
        self.assertEqual(result["messages"][0].tool_calls, [])

    @patch.object(agent_graph, "_answer_llm")
    def test_finalizer_formats_citations_from_tool_output(self, llm):
        llm.invoke.return_value = AIMessage(content='{"answer":"20 days.","citations":["S1","S9"]}')
        messages = [
            HumanMessage(content="How many leave days?"),
            ToolMessage(name="search_knowledge_base", tool_call_id="call-1", content='{"status":"relevant","sources":[{"source_id":"S1","chunk_id":17,"source":"policy.txt"}],"evidence":[]}'),
        ]
        result = agent_graph.finalize_answer({"messages": messages})
        answer, citations = agent_graph.validate_citations(messages + result["messages"], result["messages"][0].content)
        self.assertEqual(answer, "20 days. [S1]")
        self.assertEqual(citations[0]["chunk_id"], 17)

    @patch.object(agent_graph, "_answer_llm")
    def test_finalizer_rejects_non_source_ids(self, llm):
        llm.invoke.return_value = AIMessage(content='{"answer":"6000","citations":["calculator"]}')
        messages = [
            HumanMessage(content="125 multiplied by 48?"),
            ToolMessage(name="calculator", tool_call_id="call-1", content="6000"),
        ]
        result = agent_graph.finalize_answer({"messages": messages})
        self.assertEqual(result["messages"][0].content, "6000")

    def test_finalizer_abstains_without_relevant_internal_evidence(self):
        messages = [
            HumanMessage(content="Can unused leaves be converted into cash?"),
            ToolMessage(name="search_knowledge_base", tool_call_id="call-1", content='{"status":"low_relevance","sources":[],"evidence":[]}'),
        ]
        result = agent_graph.finalize_answer({"messages": messages})
        self.assertEqual(result["messages"][0].content, "The available documents do not provide information about this request.")


if __name__ == "__main__":
    unittest.main()
