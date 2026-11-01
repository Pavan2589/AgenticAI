
import asyncio
import json
import os

import ollama

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters
)


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

MODEL = "qwen3:4b"
MAX_TOOL_ITERATIONS = 8

GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GitHub token not found in .env")


# ==================================================
# GITHUB MCP SERVER CONFIGURATION
# ==================================================

server_params = StdioServerParameters(
    command="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "-e", "GITHUB_READ_ONLY=1",
        "-e", "GITHUB_TOOLSETS=repos,issues,pull_requests",
        "ghcr.io/github/github-mcp-server"
    ],
    env=os.environ.copy()
)


# ==================================================
# MCP TOOLS -> OLLAMA TOOLS CONVERSION
# ==================================================

def convert_mcp_tools_to_ollama(mcp_tools):

    ollama_tools = []

    for tool in mcp_tools:

        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema
            }
        })

    return ollama_tools


# ==================================================
# MCP RESULT -> TEXT
# ==================================================

def format_tool_result(result):

    output = []

    for item in result.content:

        if hasattr(item, "text"):
            output.append(item.text)
        else:
            output.append(str(item))

    content = "\n".join(output)

    if result.isError:
        return f"Tool execution failed:\n{content}"

    return content or "Tool returned no text content."


# ==================================================
# AGENT LOOP
# ==================================================

async def run_agent(session, ollama_tools):

    messages = [
        {
            "role": "system",
            "content": """
You are a GitHub assistant connected to GitHub through MCP.

You can use the available GitHub tools to answer questions.

Instructions:
1. Use GitHub tools when the user asks for live GitHub information.
2. Select the appropriate tool based on its description.
3. Provide all required arguments.
4. Never invent repository names, issues, commits or results.
5. Use tool results as the source of truth.
6. Explain the results clearly to the user.
7. If a tool fails, explain the error instead of inventing an answer.
8. You have read-only access. Do not claim to modify GitHub.
"""
        }
    ]

    print("\nGitHub Agent is ready!")
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input
        })

        # ------------------------------------------
        # Ask Qwen to respond or select tools
        # ------------------------------------------

        for iteration in range(MAX_TOOL_ITERATIONS):

            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=ollama_tools,
                think=False,
                keep_alive="10m",
                options={
                    "temperature": 0,
                    "num_ctx": 4096
                }
            )

            assistant_message = response.message

            # Add Qwen's response to conversation history
            messages.append(assistant_message)

            # --------------------------------------
            # Check whether Qwen requested a tool
            # --------------------------------------

            tool_calls = assistant_message.tool_calls

            if not tool_calls:

                print(f"\nAssistant: {assistant_message.content}\n")
                break

            # --------------------------------------
            # Execute requested MCP tools
            # --------------------------------------

            for tool_call in tool_calls:

                tool_name = tool_call.function.name

                arguments = tool_call.function.arguments or {}

                print(f"\nCalling MCP tool: {tool_name}")
                print(
                    "Arguments:",
                    json.dumps(arguments, indent=2)
                )

                try:

                    result = await session.call_tool(
                        tool_name,
                        arguments=arguments
                    )

                    tool_output = format_tool_result(result)

                except Exception as exc:

                    tool_output = (
                        f"Error executing {tool_name}: {str(exc)}"
                    )

                print(f"\nTool result:\n{tool_output}\n")

                # Return tool result to Qwen
                messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": tool_output
                })

        else:
            print(
                "\nMaximum tool iterations reached. "
                "Please try a simpler request."
            )


# ==================================================
# MAIN APPLICATION
# ==================================================

async def main():

    print("Starting GitHub MCP Agent...")

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize MCP session
            await session.initialize()

            print("Connected to GitHub MCP server!")

            # Dynamically discover tools
            result = await session.list_tools()

            mcp_tools = result.tools

            print(f"Discovered {len(mcp_tools)} GitHub tools.")

            # Convert MCP schemas to Ollama format
            ollama_tools = convert_mcp_tools_to_ollama(
                mcp_tools
            )

            print(f"Registered {len(ollama_tools)} tools with Qwen.")

            # Start agent
            await run_agent(session, ollama_tools)


if __name__ == "__main__":
    asyncio.run(main())
