
import asyncio
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters
)


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError(
        "GitHub token not found. Check your .env file."
    )


# --------------------------------------------------
# 2. Configure GitHub MCP server
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Connect to MCP server
# --------------------------------------------------

async def main():

    print("Connecting to GitHub MCP server...")

    async with stdio_client(server_params) as (read, write):

        print("STDIO connection established!")

        # Create MCP client session
        async with ClientSession(read, write) as session:

            # Initialize MCP handshake
            await session.initialize()

            print("\nConnected to GitHub MCP server!")

            # --------------------------------------------------
            # 4. Discover available tools
            # --------------------------------------------------

            result = await session.list_tools()

            tools = result.tools

            print(f"\nDiscovered {len(tools)} tools:\n")

            for index, tool in enumerate(tools, start=1):

                print(f"\n{'=' * 70}")
                print(f"Tool {index}: {tool.name}")
                print(f"{'=' * 70}")

                print("\nDescription:")
                print(tool.description or "No description available")

                print("\nInput Schema:")
                print(
                    json.dumps(
                        tool.input_schema,
                        indent=4
                    )
                )

            print("\nFinished discovering GitHub tools!")


# --------------------------------------------------
# 5. Run application
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())

