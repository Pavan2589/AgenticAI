from mcp.server import MCPServer

mcp = MCPServer("Personal Tools")

@mcp.tool()
def calculator(a: float, operation: str, b: float) -> float:
    """Perform basic arithmetic operations."""

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    raise ValueError("Unsupported operation")


if __name__ == "__main__":
    mcp.run()