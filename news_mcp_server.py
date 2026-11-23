
#Local Streamable HTTP MCP news example, with clearly synthetic responses.

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("DemoNews")


@mcp.tool()
def market_news(query: str) -> str:
    """Return a synthetic example headline; does NOT retrieve actual news."""
    if not query.strip():
        raise ToolError("query must not be blank")
    return f"SYNTHETIC DEMO: Example news search for {query!r}. No live articles were fetched."


@mcp.tool()
def market_sentiment(symbol: str) -> str:
    """Return fictional market sentiment; NOT an investment recommendation."""
    if not symbol.strip():
        raise ToolError("symbol must not be blank")
    return f"SYNTHETIC DEMO: {symbol.upper()} sentiment unavailable without real market data."


if __name__ == "__main__":
    # MCP SDK 2.x: host and port are run() arguments; endpoint is /mcp.
    # Kept on loopback for local educational testing, with no public exposure.
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
