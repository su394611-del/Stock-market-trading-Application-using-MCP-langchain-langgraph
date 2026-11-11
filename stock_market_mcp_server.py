#Local STDIO MCP stock demo server. All prices and trades are simulated.
from mcp.server import MCPServer  # official MCP SDK 2.x

from MCP_Langchain_Langraph_Tools import fake_price, simulated_trade

mcp = MCPServer("DemoStockMarket")


@mcp.tool()
def stock_price(name: str) -> float:
    """Return an illustrative static price in USD, NOT a live stock quote."""
    return fake_price(name)


@mcp.tool()
def buy_stock(name: str, qty: int) -> str:
    """SIMULATE a buy order only; no real transaction is performed."""
    return simulated_trade("buy", name, qty)


@mcp.tool()
def sell_stock(name: str, qty: int) -> str:
    """SIMULATE a sell order only; no real transaction is performed."""
    return simulated_trade("sell", name, qty)


if __name__ == "__main__":
    # STDIO is the transport protocol; do not print to stdout here.
    mcp.run(transport="stdio")
