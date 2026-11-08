import json
import math
import operator
import time

from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import ollama


# ---------------- Configuration ----------------

MODEL = "qwen3:4b"

MAX_MODEL_ROUNDS = 5
MAX_TOOL_CALLS = 8
API_TIMEOUT_SECONDS = 8
MAX_API_ATTEMPTS = 3


# ---------------- Tool 1: Calculator ----------------

def calculator(a, operation, b):
    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }

    if operation not in operations:
        return {"error": "Unsupported arithmetic operation."}

    try:
        a = float(a)
        b = float(b)

        if operation == "divide" and b == 0:
            return {"error": "Division by zero is not allowed."}

        result = operations[operation](a, b)

        if not math.isfinite(result):
            return {"error": "Result is not finite."}

        return {"result": result}

    except (ValueError, TypeError, OverflowError):
        return {"error": "Invalid numeric arguments."}


# ---------------- Tool 2: Country API ----------------

def country_lookup(country):
    if not isinstance(country, str):
        return {"error": "Country name must be text."}

    country = country.strip()

    if not country or len(country) > 100:
        return {"error": "Invalid country name."}

    url = (
        "https://countries.dev/name/"
        + quote(country, safe="")
        + "?fields=name,capital,population,region"
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

            if not isinstance(data, list):
                return {"error": "Unexpected API response format."}

            match = next(
                (
                    item for item in data
                    if isinstance(item, dict)
                    and isinstance(item.get("name"), str)
                    and item["name"].casefold() == country.casefold()
                ),
                None,
            )

            if match is None:
                return {"error": f"No exact country match for '{country}'."}

            return {
                "name": match.get("name"),
                "capital": match.get("capital"),
                "population": match.get("population"),
                "region": match.get("region"),
            }

        except HTTPError as exc:
            if exc.code == 404:
                return {"error": f"Country '{country}' was not found."}

            if exc.code not in (429, 500, 502, 503, 504):
                return {"error": f"Country API returned HTTP {exc.code}."}

            if attempt == MAX_API_ATTEMPTS - 1:
                return {"error": "Country API failed after retries."}

        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            if attempt == MAX_API_ATTEMPTS - 1:
                return {"error": f"Country API request failed: {exc}"}

        time.sleep(2 ** attempt)

    return {"error": "Country lookup failed."}


# ---------------- Tool definitions ----------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic.",
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
            "name": "country_lookup",
            "description": (
                "Look up a country's capital, population, and region "
                "using the public countries.dev API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                },
                "required": ["country"],
            },
        },
    },
]


# ---------------- Tool dispatcher ----------------

def execute_tool(name, arguments):
    """Execute only registered tools."""

    available_tools = {
        "calculator": calculator,
        "country_lookup": country_lookup,
    }

    if name not in available_tools:
        return {"error": f"Tool '{name}' is not allowed."}

    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be an object."}

    try:
        return available_tools[name](**arguments)
    except (TypeError, ValueError, OverflowError) as exc:
        return {"error": f"Invalid tool arguments: {exc}"}


# ---------------- Agent loop ----------------

def run_agent(user_input):
    messages = [
        {
            "role": "system",
            "content": """
You are a helpful assistant that can use tools.

Use country_lookup for country information.
Use calculator for arithmetic.
For multi-step tasks, use the result from one tool
to decide what to do next.
Never invent tool results.
After you have enough information, return a final answer.
""",
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    total_tool_calls = 0

    for round_number in range(1, MAX_MODEL_ROUNDS + 1):
        print(f"\n--- Model round {round_number} ---")

        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                think=False,
                options={"temperature": 0},
            )
        except Exception as exc:
            return f"Model request failed: {exc}"

        assistant_message = response.message
        tool_calls = assistant_message.tool_calls or []

        # Preserve the model response and any requested tool calls.
        messages.append(assistant_message)

        # No tool request means the model is finished.
        if not tool_calls:
            print("[Agent terminated: final answer received]")
            return assistant_message.content or "No answer generated."

        for tool_call in tool_calls:
            # Enforce the total tool-call limit.
            if total_tool_calls >= MAX_TOOL_CALLS:
                print("[Agent terminated: tool-call limit reached]")
                return "Stopped safely: maximum tool-call limit reached."

            name = tool_call.function.name
            arguments = tool_call.function.arguments

            total_tool_calls += 1

            print(f"[Tool call {total_tool_calls}/{MAX_TOOL_CALLS}]")
            print(f"[Tool requested: {name}]")
            print(f"[Arguments: {arguments}]")

            # Execute the tool.
            result = execute_tool(name, arguments)

            print(f"[Observation: {result}]")

            # Feed the observation back to the model.
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result),
            })

    print("[Agent terminated: model-round limit reached]")
    return (
        "Stopped safely: maximum model rounds reached "
        "before a final answer was produced."
    )


# ---------------- Main program ----------------

def main():
    print("Day 5: Controlled Agent Loop")
    print(f"Model: {MODEL}")
    print(f"Max model rounds: {MAX_MODEL_ROUNDS}")
    print(f"Max tool calls: {MAX_TOOL_CALLS}")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not user_input:
                continue

            answer = run_agent(user_input)
            print(f"\nAssistant: {answer}")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
