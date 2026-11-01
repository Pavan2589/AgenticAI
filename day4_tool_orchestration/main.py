
import json
import math
import operator
import sqlite3
import time

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import ollama


# ============================================================
# Configuration
# ============================================================

MODEL = "qwen3:4b"

MAX_TOOL_ROUNDS = 5
MAX_API_ATTEMPTS = 3
API_TIMEOUT_SECONDS = 8

DB_PATH = Path(__file__).with_name("day4_tasks.db")


# ============================================================
# 1. Initialize demo database
# ============================================================

def initialize_demo_database():
    """Create a demo database and seed it with tasks if empty."""

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK(status IN ('pending', 'in_progress', 'completed'))
            )
        """)

        count = connection.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0]

        if count == 0:
            connection.executemany(
                "INSERT INTO tasks (title, status) VALUES (?, ?)",
                [
                    ("Build API client", "completed"),
                    ("Add retry handling", "pending"),
                    ("Test database tool", "in_progress"),
                    ("Write project documentation", "pending"),
                ],
            )


# ============================================================
# 2. Calculator tool
# ============================================================

def calculator(a, operation, b):
    """Perform a basic arithmetic operation."""

    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }

    if operation not in operations:
        return {"error": f"Unsupported operation: {operation}"}

    if operation == "divide" and b == 0:
        return {"error": "Division by zero is not allowed."}

    try:
        a = float(a)
        b = float(b)
        result = operations[operation](a, b)

        if not math.isfinite(result):
            return {"error": "The result is not finite."}

        return {"result": result}

    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return {"error": "Invalid numbers supplied to calculator."}


# ============================================================
# 3. Public country API tool
# ============================================================

def country_lookup(country):
    """
    Look up a country using countries.dev.
    No API key is required.
    """

    if not isinstance(country, str):
        return {"error": "Country name must be text."}

    country = country.strip()

    if not country or len(country) > 100:
        return {"error": "Country name must be between 1 and 100 characters."}

    url = (
        "https://countries.dev/name/"
        f"{quote(country)}"
        "?fields=name,capital,population,region"
    )

    for attempt in range(MAX_API_ATTEMPTS):
        try:
            request = Request(
                url,
                headers={"Accept": "application/json"},
            )

            with urlopen(
                request,
                timeout=API_TIMEOUT_SECONDS,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

            if not isinstance(data, list) or not data:
                return {"error": f"No country records found for '{country}'."}

            # Avoid selecting a partial match such as
            # British Indian Ocean Territory for "India".
            match = next(
                (
                    item
                    for item in data
                    if isinstance(item, dict)
                    and item.get("name", "").casefold()
                    == country.casefold()
                ),
                None,
            )

            if match is None:
                return {
                    "error": f"No exact country match found for '{country}'."
                }

            return {
                "name": match.get("name"),
                "capital": match.get("capital"),
                "population": match.get("population"),
                "region": match.get("region"),
            }

        except HTTPError as exc:
            if exc.code == 404:
                return {"error": f"No country found for '{country}'."}

            retryable = exc.code in (429, 500, 502, 503, 504)

            if not retryable:
                return {
                    "error": f"Country API returned HTTP {exc.code}."
                }

            if attempt == MAX_API_ATTEMPTS - 1:
                return {
                    "error": "Country API failed after multiple retries."
                }

        except (
            URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            if attempt == MAX_API_ATTEMPTS - 1:
                return {
                    "error": f"Country API request failed: {exc}"
                }

        # Exponential backoff: 1 second, then 2 seconds.
        time.sleep(2 ** attempt)

    return {"error": "Country API request failed."}


# ============================================================
# 4. Read-only SQLite database tool
# ============================================================

def query_tasks(action, status="", task_id=0):
    """
    Read tasks using approved queries only.
    No INSERT, UPDATE, DELETE, or arbitrary SQL is permitted.
    """

    allowed_actions = {"list", "by_status", "by_id"}

    if action not in allowed_actions:
        return {"error": "Invalid action. Use list, by_status, or by_id."}

    if action == "by_status":
        allowed_statuses = {"pending", "in_progress", "completed"}

        if status not in allowed_statuses:
            return {
                "error": (
                    "Invalid status. Use pending, in_progress, or completed."
                )
            }

    if action == "by_id":
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            return {"error": "Task ID must be an integer."}

        if task_id <= 0:
            return {"error": "Task ID must be positive."}

    try:
        # Open the database in read-only mode.
        db_uri = DB_PATH.resolve().as_uri() + "?mode=ro"

        with sqlite3.connect(
            db_uri,
            uri=True,
            timeout=3,
        ) as connection:
            connection.row_factory = sqlite3.Row

            if action == "list":
                rows = connection.execute(
                    "SELECT id, title, status FROM tasks ORDER BY id"
                ).fetchall()

            elif action == "by_status":
                rows = connection.execute(
                    """
                    SELECT id, title, status
                    FROM tasks
                    WHERE status = ?
                    ORDER BY id
                    """,
                    (status,),
                ).fetchall()

            else:
                rows = connection.execute(
                    """
                    SELECT id, title, status
                    FROM tasks
                    WHERE id = ?
                    """,
                    (task_id,),
                ).fetchall()

        tasks = [dict(row) for row in rows]

        return {
            "tasks": tasks,
            "count": len(tasks),
        }

    except sqlite3.Error as exc:
        return {"error": f"Database query failed: {exc}"}


# ============================================================
# 5. Tool definitions for Qwen
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["a", "operation", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "country_lookup",
            "description": (
                "Look up a country's capital, population, and region "
                "using a public API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Exact country name, e.g. India",
                    }
                },
                "required": ["country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_tasks",
            "description": (
                "Read tasks from the database. Use list to retrieve all tasks, "
                "by_status to filter by status, or by_id to find one task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "by_status", "by_id"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "Required when action is by_status",
                    },
                    "task_id": {
                        "type": "integer",
                        "description": "Required when action is by_id",
                    },
                },
                "required": ["action"],
            },
        },
    },
]


# ============================================================
# 6. Tool dispatcher
# ============================================================

def execute_tool(name, arguments):
    """Validate tool requests and execute only approved functions."""

    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be a JSON object."}

    available_tools = {
        "calculator": calculator,
        "country_lookup": country_lookup,
        "query_tasks": query_tasks,
    }

    tool = available_tools.get(name)

    if tool is None:
        return {"error": f"Tool '{name}' is not allowed."}

    try:
        return tool(**arguments)

    except TypeError:
        return {
            "error": f"Invalid arguments supplied to tool '{name}'."
        }

    except (ValueError, OverflowError) as exc:
        return {"error": str(exc)}


# ============================================================
# 7. Agent orchestration loop
# ============================================================

def ask_assistant(user_text):
    """Allow Qwen to request tools and use their results."""

    messages = [
        {
            "role": "system",
            "content": """
You are a helpful assistant with access to three tools.

Rules:
- Use tools when the user asks for live country information,
  arithmetic, or task database information.
- Never invent API or database results.
- For questions requiring multiple steps, call the appropriate tools
  in sequence and use the actual results returned by previous tools.
- For calculating a percentage of a population, first call country_lookup,
  then call calculator using the returned population.
- Explain errors honestly.
- The task database is read-only.
- After receiving tool results, provide a concise final answer.
""",
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]

    for round_number in range(MAX_TOOL_ROUNDS):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                think=False,
                options={"temperature": 0},
            )

        except Exception as exc:
            return f"LLM request failed: {exc}"

        assistant_message = response.message
        tool_calls = assistant_message.tool_calls or []

        # Preserve the assistant message, including its tool calls.
        messages.append(assistant_message)

        # No tool calls means the assistant has completed its response.
        if not tool_calls:
            return assistant_message.content or "No response was generated."

        # Execute every tool call requested in this round.
        for tool_call in tool_calls:
            function = tool_call.function
            name = function.name
            arguments = function.arguments

            print(f"\n[Tool requested: {name}]")
            print(f"[Arguments: {arguments}]")

            result = execute_tool(name, arguments)

            print(f"[Tool result: {result}]")

            # Associate the result with the requested tool call.
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result),
                }
            )

    return (
        "I reached the maximum number of tool rounds. "
        "Please try a simpler request."
    )


# ============================================================
# 8. CLI entry point
# ============================================================

def main():
    initialize_demo_database()

    print("Day 4: Tool Orchestration Assistant")
    print("Tools: calculator, country_lookup, query_tasks")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_text = input("\nYou: ").strip()

            if user_text.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not user_text:
                continue

            answer = ask_assistant(user_text)
            print(f"\nAssistant: {answer}")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
