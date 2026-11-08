import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from state import AgentState
from tools import tools


# 1. Initialize local LLM
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

# 2. Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# 3. Define chatbot node
def chatbot(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])

    print("\n[CHATBOT NODE]")
    print("Tool calls:", response.tool_calls)

    return {
        "messages": [response]
    }


# 4. Define tool execution node
def tool_node(state: AgentState):
    print("\n[TOOL NODE] Executing tools...")

    return ToolNode(tools).invoke(state)


# 5. Build the graph
builder = StateGraph(AgentState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)


# 6. Define graph flow
builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

builder.add_edge("tools", "chatbot")

# 7. Create SQLite database connection

conn = sqlite3.connect(
    "checkpoints.sqlite",
    check_same_thread=False
)

# 8. Initialize SQLite checkpointer
memory = SqliteSaver(conn)
memory.setup()

# 9. Compile graph with persistent checkpointing
graph = builder.compile(checkpointer=memory)