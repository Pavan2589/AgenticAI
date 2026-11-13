import sqlite3
import json
import uuid
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import LLM_MODEL
from prompts import SYSTEM_PROMPT, TOOL_SELECTION_PROMPT, GROUNDED_RESPONSE_PROMPT
from state import AgentState
from tools import tools

# Disable Qwen3's verbose reasoning mode and cap generated tokens. Without
# these controls local CPU tool-calling can spend minutes reasoning before
# emitting the structured tool call.
_llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    reasoning=False,
    num_ctx=8192,
    num_predict=512,
)
_llm_with_tools = _llm.bind_tools(tools)
_answer_llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    reasoning=False,
    format="json",
    num_ctx=4096,
    num_predict=300,
)
_tool_selector = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    reasoning=False,
    format="json",
    num_ctx=4096,
    num_predict=180,
)


def chatbot(state: AgentState):
    messages = state["messages"]
    last_human_index = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if isinstance(messages[index], HumanMessage)),
        -1,
    )
    current_turn = messages[last_human_index + 1:]
    question = str(messages[last_human_index].content) if last_human_index >= 0 else ""
    prior_results = [
        {"tool": message.name, "result": message.content}
        for message in current_turn if isinstance(message, ToolMessage)
    ]
    pending_calls = {}
    executed_calls = set()
    for message in current_turn:
        if isinstance(message, AIMessage):
            for prior_call in message.tool_calls:
                pending_calls[prior_call["id"]] = (
                    prior_call["name"],
                    json.dumps(prior_call.get("args", {}), sort_keys=True),
                )
        elif isinstance(message, ToolMessage):
            signature = pending_calls.pop(message.tool_call_id, None)
            if signature:
                executed_calls.add(signature)
    if len(prior_results) >= 6:
        return {"messages": [AIMessage(content="", tool_calls=[])]}
    prompt = (
        f"User request: {question}\n\nCurrent-turn tool results:\n"
        + (json.dumps(prior_results, ensure_ascii=False) if prior_results else "None")
    )
    response = _tool_selector.invoke([
        SystemMessage(content=TOOL_SELECTION_PROMPT),
        HumanMessage(content=prompt),
    ])
    try:
        plan = json.loads(response.content)
    except (TypeError, json.JSONDecodeError):
        # Keep Part 13 native binding as a fallback if constrained planning fails.
        fallback_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        fallback = _llm_with_tools.invoke(fallback_messages)
        return {"messages": [AIMessage(content="", tool_calls=fallback.tool_calls,
                                       invalid_tool_calls=fallback.invalid_tool_calls,
                                       id=fallback.id)]}
    calls = plan.get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError("Tool planner JSON must contain a 'tool_calls' list.")
    available = {registered.name: registered for registered in tools}
    tool_calls = []
    planned_signatures = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("name") not in available:
            raise ValueError(f"Tool planner selected an unknown tool: {call!r}")
        if not isinstance(call.get("args"), dict):
            raise ValueError(f"Tool planner arguments must be an object: {call!r}")
        signature = (call["name"], json.dumps(call["args"], sort_keys=True))
        if signature in executed_calls or signature in planned_signatures:
            continue
        planned_signatures.add(signature)
        tool_calls.append({
            "name": call["name"], "args": call["args"],
            "id": f"call_{uuid.uuid4().hex}", "type": "tool_call",
        })
    # Standard LangChain tool-call objects preserve tools_condition and ToolNode.
    return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}


def finalize_answer(state: AgentState):
    """Generate a concise JSON-constrained answer after tool execution."""
    messages = state["messages"]
    last_human_index = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if isinstance(messages[index], HumanMessage)),
        -1,
    )
    current_turn = messages[last_human_index + 1:]
    question = str(messages[last_human_index].content) if last_human_index >= 0 else ""
    tool_results = []
    knowledge_status = None
    for message in current_turn:
        if isinstance(message, ToolMessage):
            if message.name == "search_knowledge_base":
                try:
                    payload = json.loads(message.content)
                except (TypeError, json.JSONDecodeError):
                    tool_results.append(f"Knowledge search returned an unreadable result: {message.content[:500]}")
                    continue
                knowledge_status = payload.get("status")
                if payload.get("status") == "relevant":
                    for evidence in payload.get("evidence", []):
                        metadata = evidence.get("effective_date")
                        version = evidence.get("document_version")
                        details = ", ".join(value for value in (f"effective {metadata}" if metadata else "", f"version {version}" if version else "") if value)
                        label = f"[{evidence.get('source_id')}] {evidence.get('source')}"
                        if details:
                            label += f" ({details})"
                        tool_results.append(f"{label}\n{evidence.get('content', '')}")
                else:
                    tool_results.append(
                        f"Knowledge search status: {payload.get('status')}. "
                        f"{payload.get('message', 'No usable document evidence was returned.')}"
                    )
            else:
                tool_results.append(f"{message.name}: {message.content}")

    if len(tool_results) == 1 and knowledge_status == "error":
        return {"messages": [AIMessage(content="Knowledge search is temporarily unavailable, so I can't verify this from the internal documents.")]}
    if len(tool_results) == 1 and knowledge_status in {"low_relevance", "no_results"}:
        return {"messages": [AIMessage(content="The available documents do not provide information about this request.")]}

    response = _answer_llm.invoke([
        SystemMessage(content=GROUNDED_RESPONSE_PROMPT),
        HumanMessage(content=(
            f"Question: {question}\n\nTool results:\n"
            + ("\n\n".join(tool_results) if tool_results else "No tools were needed for this general question.")
        )),
    ])
    try:
        data = json.loads(response.content)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Grounded answer model returned invalid JSON: {response.content}") from error
    if not isinstance(data.get("answer"), str) or not isinstance(data.get("citations"), list):
        raise ValueError("Grounded answer response must contain string 'answer' and list 'citations'.")
    allowed_source_ids = set()
    for message in current_turn:
        if not isinstance(message, ToolMessage) or message.name != "search_knowledge_base":
            continue
        try:
            search_result = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if search_result.get("status") == "relevant":
            allowed_source_ids.update(
                source.get("source_id") for source in search_result.get("sources", [])
                if isinstance(source.get("source_id"), str)
            )
    citation_text = " ".join(
        f"[{source_id}]" for source_id in dict.fromkeys(data["citations"])
        if isinstance(source_id, str) and source_id in allowed_source_ids
    )
    answer = data["answer"].strip()
    return {"messages": [AIMessage(content=f"{answer} {citation_text}".strip())]}


builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_node("finalize", finalize_answer)
builder.add_edge(START, "chatbot")
builder.add_conditional_edges(
    "chatbot", tools_condition, {"tools": "tools", "__end__": "finalize"}
)
# Keep the Part 13 tool loop so the model can decide whether a first tool
# result requires another tool (for example, retrieve policy, then calculate).
builder.add_edge("tools", "chatbot")
builder.add_edge("finalize", END)

_checkpoint_path = Path(__file__).parent / "checkpoints.sqlite"
_connection = sqlite3.connect(str(_checkpoint_path), check_same_thread=False)
_checkpointer = SqliteSaver(_connection)
_checkpointer.setup()
graph = builder.compile(checkpointer=_checkpointer)


def validate_citations(messages, answer):
    """Keep only [S#] citations present in actual search tool ToolMessages."""
    import json
    import re
    sources = {}
    relevant_evidence_seen = False
    last_human_index = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if isinstance(messages[index], HumanMessage)),
        -1,
    )
    current_turn = messages[last_human_index + 1:] if last_human_index >= 0 else messages
    for message in current_turn:
        if not isinstance(message, ToolMessage) or message.name != "search_knowledge_base":
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if payload.get("status") == "relevant":
            relevant_evidence_seen = True
            for source in payload.get("sources", []):
                if isinstance(source.get("source_id"), str):
                    sources[source["source_id"]] = source
    cited = re.findall(r"\[(S\d+)\]", answer)
    valid = list(dict.fromkeys(c for c in cited if c in sources))
    cleaned = re.sub(r"\[(S\d+)\]", lambda m: m.group(0) if m.group(1) in sources else "", answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    if relevant_evidence_seen and not valid:
        cleaned = "I couldn't verify citations for the document-based answer, so I can't present it as grounded."
    return cleaned, [sources[source_id] for source_id in valid]
