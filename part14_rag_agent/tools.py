import ast
import operator
from datetime import date, timedelta

import pycountry
from langchain_core.tools import tool
from rag_tool import search_knowledge_base

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _calculate(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_calculate(node.right)) > 10:
            raise ValueError("Exponent too large")
        return _OPS[type(node.op)](_calculate(node.left), _calculate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calculate(node.operand))
    raise ValueError("Unsupported mathematical expression")


def safe_calculate(expression: str):
    """Parse and evaluate only the supported arithmetic AST nodes."""
    return _calculate(ast.parse(expression, mode="eval").body)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    try:
        return str(safe_calculate(expression))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"


@tool
def date_utility(days_from_today: int = 0) -> str:
    """Return today's date or a date offset by the given number of days."""
    return (date.today() + timedelta(days=days_from_today)).strftime("%A, %d %B %Y")


@tool
def country_lookup(country_name: str) -> str:
    """Look up a country's official name and ISO country codes."""
    try:
        country = pycountry.countries.lookup(country_name)
        return f"Country: {country.name}\nAlpha-2 code: {country.alpha_2}\nAlpha-3 code: {country.alpha_3}"
    except LookupError:
        return f"Country '{country_name}' not found."


tools = [calculator, date_utility, country_lookup, search_knowledge_base]
