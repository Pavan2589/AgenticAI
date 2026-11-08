
import ast
import operator
from datetime import date, timedelta

import pycountry
from langchain_core.tools import tool


# --------------------------------
# 1. Calculator
# --------------------------------

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression: str):
    """Evaluate basic arithmetic without using eval()."""

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent too large")

            return OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.operand))

        raise ValueError("Unsupported mathematical expression")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


@tool
def calculator(expression: str) -> str:
    """Perform arithmetic calculations. Input should be a mathematical expression."""

    try:
        result = safe_calculate(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# --------------------------------
# 2. Date Utility
# --------------------------------

@tool
def date_utility(days_from_today: int = 0) -> str:
    """
    Get today's date or calculate a date relative to today.
    Positive values represent future dates; negative values represent past dates.
    """

    target_date = date.today() + timedelta(days=days_from_today)

    return target_date.strftime("%A, %d %B %Y")


# --------------------------------
# 3. Country Lookup
# --------------------------------

@tool
def country_lookup(country_name: str) -> str:
    """Look up a country's official name and ISO country codes."""

    try:
        country = pycountry.countries.lookup(country_name)

        return (
            f"Country: {country.name}\n"
            f"Alpha-2 code: {country.alpha_2}\n"
            f"Alpha-3 code: {country.alpha_3}"
        )

    except LookupError:
        return f"Country '{country_name}' not found."


# --------------------------------
# Register all tools
# --------------------------------

tools = [
    calculator,
    date_utility,
    country_lookup
]
