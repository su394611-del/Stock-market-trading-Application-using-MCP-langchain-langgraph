#Local STDIO MCP stock demo server. All prices and trades are simulated.
from typing import Annotated

from mcp.server import MCPServer  # official MCP SDK 2.x
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from MCP_Langchain_Langraph_Tools import fake_price, simulated_trade

mcp = MCPServer("DemoStockMarket")


@mcp.tool()
def stock_price(name: str) -> float:
    """Return an illustrative static price in USD, NOT a live stock quote."""
    try:
        return fake_price(name)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def _trade(action: str, name: str, qty: int) -> str:
    try:
        return simulated_trade(action, name, qty)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def buy_stock(name: str, qty: Annotated[int, Field(strict=True)]) -> str:
    """SIMULATE a buy order only; no real transaction is performed."""
    return _trade("buy", name, qty)


@mcp.tool()
def sell_stock(name: str, qty: Annotated[int, Field(strict=True)]) -> str:
    """SIMULATE a sell order only; no real transaction is performed."""
    return _trade("sell", name, qty)


if __name__ == "__main__":
    # STDIO is the transport protocol; do not print to stdout here.
    mcp.run(transport="stdio")
