
import json
import operator
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import ollama


MODEL = "qwen3:4b"
MAX_TOOL_ROUNDS = 5


# -----------------------------
# 1. Actual Python tool functions
# -----------------------------

def calculator(a: float, operation: str, b: float) -> dict:
    """Perform basic arithmetic."""
    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }

    if operation not in operations:
        return {"error": "Unsupported arithmetic operation."}

    if operation == "divide" and b == 0:
        return {"error": "Division by zero is not allowed."}

    try:
        result = operations[operation](a, b)
        return {"result": result}
    except (ArithmeticError, TypeError) as exc:
        return {"error": str(exc)}


def date_utility(action: str, days: int = 0) -> dict:
    """Return the current date, weekday, or a date offset."""
    today = date.today()

    if action == "today":
        return {"date": today.isoformat()}

    if action == "weekday":
        return {
            "date": today.isoformat(),
            "weekday": today.strftime("%A"),
        }

    if action == "add_days":
        target = today + timedelta(days=days)
        return {
            "start_date": today.isoformat(),
            "days_added": days,
            "result_date": target.isoformat(),
            "weekday": target.strftime("%A"),
        }

    return {"error": "Unsupported date action."}


def country_lookup(country: str) -> dict:
    """Read basic country information from the REST Countries API."""
    url = (
        "https://restcountries.com/v3.1/name/"
        + quote(country.strip(), safe="")
        + "?fields=name,capital,population,region"
    )

    request = Request(
        url,
        headers={"User-Agent": "Day3-AgenticAI-Learning-Project"},
    )

    try:
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        if not data:
            return {"error": "No country information found."}

        # Return a concise result rather than the full API response.
        item = data[0]
        return {
            "name": item.get("name", {}).get("common"),
            "capital": item.get("capital", []),
            "population": item.get("population"),
            "region": item.get("region"),
        }

    except HTTPError as exc:
        if exc.code == 404:
            return {"error": f"No country found for '{country}'."}
        return {"error": f"Country API returned HTTP {exc.code}."}

    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"error": f"Country lookup failed: {exc}"}


# ---------------------------------
# 2. Tell the LLM which tools exist
# ---------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "b": {"type": "number"},
                },
                "required": ["a", "operation", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_utility",
            "description": (
                "Get today's date, today's weekday, or a date "
                "a given number of days from today."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["today", "weekday", "add_days"],
                    },
                    "days": {
                        "type": "integer",
                        "description": (
                            "Number of days to add; required for add_days."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "country_lookup",
            "description": (
                "Look up a country's capital, population, and region "
                "using a read-only public API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Country name, such as India or Japan.",
                    },
                },
                "required": ["country"],
            },
        },
    },
]


# ---------------------------------
# 3. Dispatch tool calls to Python
# ---------------------------------

def execute_tool(name: str, arguments: dict) -> dict:
    """Execute only the tools explicitly allowed by this application."""

    try:
        if name == "calculator":
            return calculator(**arguments)

        if name == "date_utility":
            return date_utility(**arguments)

        if name == "country_lookup":
            return country_lookup(**arguments)

        return {"error": f"Unknown tool: {name}"}

    except (TypeError, ValueError, OverflowError) as exc:
        return {"error": f"Invalid tool arguments: {exc}"}


# ---------------------------------
# 4. Let the LLM use tools in a loop
# ---------------------------------

def ask_assistant(user_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the provided tools "
                "when they are useful. Use calculator for arithmetic, "
                "date_utility for date questions, and country_lookup "
                "for country facts. Do not invent tool results. "
                "Explain errors honestly. Do not claim a tool ran "
                "unless the application returned its result."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            think=False,
            options={"temperature": 0},
        )

        # Preserve the assistant response, including any tool calls.
        messages.append(response.message)

        tool_calls = response.message.tool_calls

        # No more tools requested: this is the final assistant answer.
        if not tool_calls:
            return response.message.content or "No text response received."

        for tool_call in tool_calls:
            name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"\n[Tool requested: {name}]")
            print(f"[Arguments: {arguments}]")

            result = execute_tool(name, arguments)

            print(f"[Tool result: {result}]")

            # Return the result to the model so it can continue.
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result),
                }
            )

    return (
        "I reached the tool-call limit before producing "
        "a final answer. Please try a simpler request."
    )


def main():
    print("Day 3 Tool-Calling Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_text:
            continue

        try:
            answer = ask_assistant(user_text)
            print(f"\nAssistant: {answer}\n")

        except Exception as exc:
            print(f"\nAssistant error: {exc}\n")


if __name__ == "__main__":
    main()
