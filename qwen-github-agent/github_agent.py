
import asyncio
import json
import os
import requests

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL = "qwen2.5-coder:7b"

MAX_ITERATIONS = 5

CONTEXT_LENGTH = 4096

if not GITHUB_TOKEN:
    raise ValueError(
        "GitHub token not found. Check your .env file."
    )


# ==========================================
# OLLAMA CLIENT
# ==========================================

def ollama_chat(messages, tools):

    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": CONTEXT_LENGTH,
            "num_predict": 512
        },
        "keep_alive": "10m"
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    return response.json()["message"]


# ==========================================
# MCP TOOL CONVERSION
# ==========================================

def convert_mcp_tools(mcp_tools):

    ollama_tools = []

    for tool in mcp_tools:

        # Support MCP SDK schema attribute naming
        schema = getattr(
            tool,
            "inputSchema",
            getattr(tool, "input_schema", None)
        )

        if schema is None:
            schema = {
                "type": "object",
                "properties": {}
            }

        ollama_tools.append({

            "type": "function",

            "function": {
                "name": tool.name,

                "description": tool.description or "",

                "parameters": schema
            }
        })

    return ollama_tools


# ==========================================
# MCP RESULT EXTRACTION
# ==========================================

def extract_tool_result(result):

    output = []

    for item in result.content:

        if hasattr(item, "text"):

            output.append(item.text)

    if not output and getattr(result, "structuredContent", None):
        output.append(
            json.dumps(result.structuredContent, default=str)
        )

    content = "\n".join(output)

    if getattr(result, "isError", False):
        return "MCP TOOL ERROR: " + content

    return content or "Tool executed successfully. No content returned."


# ==========================================
# AGENT
# ==========================================

async def main():

    # Docker MCP server configuration

    server_params = StdioServerParameters(

        command="docker",

        args=[
            "run",
            "-i",
            "--rm",

            "-e",
            f"GITHUB_PERSONAL_ACCESS_TOKEN={GITHUB_TOKEN}",

            "-e",
            "GITHUB_READ_ONLY=1",

            "-e",
            "GITHUB_TOOLSETS=repos",

            "ghcr.io/github/github-mcp-server:v2.0.3"
        ]
    )

    print("\nConnecting to GitHub MCP server...")

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize MCP connection

            await session.initialize()

            print("\nMCP connection initialized!")

            # Discover available MCP tools

            tools_response = await session.list_tools()

            all_tools = tools_response.tools

            print(
                f"\nTotal MCP tools discovered: {len(all_tools)}"
            )

            # Initially expose only repository search

            mcp_tools = [

                tool for tool in all_tools

                if tool.name == "search_repositories"

            ]

            if not mcp_tools:

                raise RuntimeError(
                    "search_repositories tool not found!"
                )

            print("\nEnabled tools:")

            for tool in mcp_tools:
                print(f" - {tool.name}")

            # Convert MCP tools to Ollama format

            ollama_tools = convert_mcp_tools(mcp_tools)

            print("\nQwen GitHub Agent is ready!")

            # ==================================
            # INTERACTIVE LOOP
            # ==================================

            while True:

                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    break

                if not user_input:
                    continue

                # Fresh conversation for each request
                messages = [

                    {
                        "role": "system",

                        "content": """
You are an intelligent GitHub repository assistant.

You have access to the GitHub MCP tool:

search_repositories

IMPORTANT INSTRUCTIONS:

1. When the user asks about GitHub repositories,
   you MUST call search_repositories.

2. Never invent repository information.

3. Wait for the actual tool result before answering.

4. Use the repository search query supplied by
   the user or construct an appropriate GitHub
   search query.

5. Clearly report errors if the tool fails.

6. Never claim that you accessed information
   that was not returned by the tool.

7. This GitHub connection is READ ONLY.

Call the available tool rather than answering
from your training knowledge.
"""
                    },

                    {
                        "role": "user",
                        "content": user_input
                    }

                ]

                # ==================================
                # TOOL CALLING LOOP
                # ==================================

                for iteration in range(MAX_ITERATIONS):

                    print(
                        f"\n[Qwen thinking... "
                        f"Iteration {iteration + 1}]"
                    )

                    try:

                        assistant_message = ollama_chat(
                            messages,
                            ollama_tools
                        )

                    except requests.exceptions.RequestException as error:

                        print("\nOllama API error:", error)
                        break

                    # Add assistant response to conversation

                    messages.append(assistant_message)

                    # Extract tool calls

                    tool_calls = assistant_message.get(
                        "tool_calls", []
                    )

                    # No tool calls means final response

                    if not tool_calls:

                        answer = assistant_message.get(
                            "content", ""
                        )

                        print("\nQwen:", answer)
                        break

                    # ==================================
                    # EXECUTE MCP TOOLS
                    # ==================================

                    for tool_call in tool_calls:

                        function = tool_call["function"]

                        tool_name = function["name"]

                        arguments = function.get(
                            "arguments", {}
                        )

                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)

                        print(
                            f"\n[Calling MCP Tool: {tool_name}]"
                        )

                        print(
                            "Arguments:",
                            json.dumps(arguments, indent=2)
                        )

                        # Validate tool name

                        allowed_tools = {
                            tool.name for tool in mcp_tools
                        }

                        if tool_name not in allowed_tools:

                            tool_output = (
                                f"Error: Tool {tool_name} "
                                "is not available."
                            )

                        else:

                            try:

                                # Execute MCP tool

                                result = await session.call_tool(
                                    tool_name,
                                    arguments=arguments
                                )

                                tool_output = extract_tool_result(
                                    result
                                )

                                print(
                                    "\n[GitHub MCP result]"
                                )

                                print(
                                    tool_output[:5000]
                                )

                            except Exception as error:

                                tool_output = (
                                    f"Tool execution failed: {error}"
                                )

                        # Send MCP result back to Qwen

                        messages.append({

                            "role": "tool",

                            "tool_name": tool_name,

                            "content": tool_output

                        })

                    print(
                        "\n[Tool results sent back to Qwen]"
                    )

                else:

                    print(
                        "\nMaximum tool-call iterations reached."
                    )

            print("\nDisconnecting GitHub MCP server...")

    print("\nGitHub Agent stopped.")


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    asyncio.run(main())
