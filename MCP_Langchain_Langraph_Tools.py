# MCP_Langchain_Langraph_Tools.py --self-test
# MCP_Langchain_Langraph_Tools.py --list
# MCP_Langchain_Langraph_Tools.py --topic all
# MCP_Langchain_Langraph_Tools.py --topic mcp-client
# MCP_Langchain_Langraph_Tools.py --topic react --include-news
# MCP_Langchain_Langraph_Tools.py --topic tools --live  # requires OPENAI_API_KEY

#For --include-news, launch the HTTP example in another terminal:
 #   python news_mcp_server.py
from __future__ import annotations
import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence
from importlib import import_module

ROOT = Path(__file__).resolve().parent
MODEL_DEFAULT = "gpt-5-mini"
TOPICS = (
    "tools", "tool-calling", "manual-execution", "graph-basic",
    "graph-manual-tools", "graph-conditional", "graph-prebuilt",
    "tavily", "mcp-client", "mcp-graph", "react",
)

# These are intentionally fixed EXAMPLE values, NOT real-time stock quotes.
DEMO_USD_PRICES = {
    "apple": 175.0, "microsoft": 320.0, "google": 135.0,
    "alphabet": 135.0, "ibm": 145.0, "tesla": 215.0,
    "nvidia": 125.0,
}


def fake_price(name: str) -> float:
    """Deterministic USD example price; does not access any exchange."""
    name = name.strip().lower()
    if not name:
        raise ValueError("name must not be empty")
    return DEMO_USD_PRICES.get(name, 100.0)


def simulated_trade(action: str, name: str, qty: int) -> str:
    """Never executes a broker order. Used ONLY for learning tool calls."""
    if action not in ("buy", "sell"):
        raise ValueError("action must be buy or sell")
    if not name.strip() or isinstance(qty, bool) or not isinstance(qty, int) or qty < 1:
        raise ValueError("name must be supplied and qty must be a positive integer")
    return f"SIMULATION ONLY: would {action} {qty} shares of {name}. No trade was placed."


def content_text(content: Any) -> str:
    """Plain text of a message's content (a string, or LangChain/MCP content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(c.get("text", "")) if isinstance(c, dict) else str(getattr(c, "text", c))
                       for c in content if not isinstance(c, dict) or c.get("type", "text") == "text")
    return str(content)


def interpret_demo_request(text: str, allowed_tools: Sequence[str]) -> list[dict[str, Any]]:
    """Simple offline rule-based tool-call demonstration. NOT a language model."""
    allowed = set(allowed_tools)

    def resolve_tool(base_name: str) -> str | None:
        # The modern MCPAdapter prefixes MCP tools with their server names.
        if base_name in allowed:
            return base_name
        return next((n for n in sorted(allowed) if n.endswith("_" + base_name)), None)

    calls: list[dict[str, Any]] = []
    # Supports 'sell 25 Apple shares' and 'buy 10 shares of Microsoft'.
    trade_pattern = (
        r"\b(buy|purchase|sell|sale)\s+(\d+)\s+"
        r"(?:(?:shares?|stocks?)\s+(?:of\s+)?)?([A-Za-z][A-Za-z.-]*)"
    )
    for match in re.finditer(trade_pattern, text, re.IGNORECASE):
        verb, quantity, company = match.groups()
        tool_name = "sell_stock" if verb.lower() in ("sell", "sale") else "buy_stock"
        actual_tool = resolve_tool(tool_name)
        if actual_tool is not None:
            calls.append({"name": actual_tool, "args": {"name": company, "qty": int(quantity)},
                          "id": f"demo_call_{len(calls) + 1}", "type": "tool_call"})
    if calls:
        return calls
    price_tool = resolve_tool("stock_price")
    if re.search(r"\b(price|quote)\b", text, re.IGNORECASE) and price_tool is not None:
        for company in DEMO_USD_PRICES:
            if re.search(rf"\b{re.escape(company)}\b", text, re.IGNORECASE):
                calls.append({"name": price_tool, "args": {"name": company.title()},
                              "id": "demo_call_1", "type": "tool_call"})
                break
    if re.search(r"\b(sentiment|sentiments|news|headlines)\b", text, re.IGNORECASE):
        sentiment_tool = resolve_tool("market_sentiment")
        news_tool = resolve_tool("market_news")
        if sentiment_tool is not None:
            company = next((k for k in DEMO_USD_PRICES if re.search(rf"\b{k}\b", text, re.I)), "market")
            calls.append({"name": sentiment_tool, "args": {"symbol": company.upper()},
                          "id": f"demo_call_{len(calls) + 1}", "type": "tool_call"})
        elif news_tool is not None:
            calls.append({"name": news_tool, "args": {"query": text},
                          "id": f"demo_call_{len(calls) + 1}", "type": "tool_call"})
    return calls


def self_test() -> None:
    """Test independent logic even if no third-party packages are installed."""
    assert fake_price("Apple") == 175.0
    assert "No trade was placed" in simulated_trade("sell", "Apple", 3)
    for bad in (0, -1):
        try:
            simulated_trade("buy", "Apple", bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid quantity accepted")
    t = interpret_demo_request("Sell 25 Apple shares and purchase 10 Microsoft stocks",
                               ["sell_stock", "buy_stock"])
    assert [x["name"] for x in t] == ["sell_stock", "buy_stock"], t
    assert [x["args"]["qty"] for x in t] == [25, 10]
    assert not interpret_demo_request("Explain the P/E ratio", ["stock_price"])
    q = interpret_demo_request("What is the price of IBM stocks?", ["stock_price"])
    assert len(q) == 1 and q[0]["args"]["name"] == "Ibm"
    for filename in ("stock_market_mcp_server.py", "news_mcp_server.py", "requirements.txt"):
        assert (ROOT / filename).is_file(), f"Missing package file: {filename}"
    print("PASS: pure-Python parsing, simulated tools, safety checks, package files")


def make_tools():
    """Create LangChain tools. Requires dependencies from requirements.txt."""
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    class TradeInput(BaseModel):
        name: str
        qty: int = Field(strict=True, description="A positive number of shares")

    @tool
    def stock_price(name: str) -> float:
        """Look up an illustrative, fixed stock price in USD (NOT live data)."""
        return fake_price(name)

    @tool(args_schema=TradeInput)
    def buy_stock(name: str, qty: int) -> str:
        """SIMULATE buying qty shares; does NOT execute a brokerage transaction."""
        return simulated_trade("buy", name, qty)

    @tool(args_schema=TradeInput)
    def sell_stock(name: str, qty: int) -> str:
        """SIMULATE selling qty shares; does NOT execute a brokerage transaction."""
        return simulated_trade("sell", name, qty)

    return [stock_price, buy_stock, sell_stock]


class DemoToolCallingModel:
    """Offline AIMessage producer to teach tool orchestration without an API key.

    This is a deterministic replacement for an LLM and does not do natural
    language reasoning. It supports the same .bind_tools/.invoke/.ainvoke shape.
    """

    def __init__(self, tools: Sequence[Any] = ()) -> None:
        self.tools = list(tools)

    def bind_tools(self, tools: Sequence[Any]):
        return DemoToolCallingModel(tools)

    def invoke(self, messages: Any):
        from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
        if isinstance(messages, str):
            text = messages
            msgs = []
            last = None
        else:
            msgs = [messages] if isinstance(messages, HumanMessage) else list(messages)
            last = msgs[-1] if msgs else None
            text = next((str(m.content) for m in reversed(msgs)
                         if isinstance(m, HumanMessage)), "")
        if isinstance(last, ToolMessage):
            results = []
            for message in reversed(msgs):
                if not isinstance(message, ToolMessage):
                    break
                results.append(message)
            failed = any(message.status == "error" for message in results)
            label = "Demo tool execution reported an error: " if failed else "Demo tool execution complete: "
            details = " | ".join(content_text(message.content) for message in reversed(results))
            return AIMessage(content=label + details)
        if "p/e" in text.lower() or "pe ratio" in text.lower():
            return AIMessage(content="P/E ratio = stock price divided by earnings per share (demo).")
        if "should i" in text.lower() or "recommendation" in text.lower():
            return AIMessage(content="Demo only: no real-time analysis or investment recommendation is provided.")
        calls = interpret_demo_request(text, [tool.name for tool in self.tools])
        if calls:
            prior_calls = sum(len(message.tool_calls) for message in msgs if isinstance(message, AIMessage))
            for index, call in enumerate(calls, prior_calls + 1):
                call["id"] = f"demo_call_{index}"
            return AIMessage(content="", tool_calls=calls)
        return AIMessage(content="Demo mode: no tool call selected for this question.")

    async def ainvoke(self, messages: Any):
        return self.invoke(messages)


def choose_llm(*, live: bool = False, model_name: str = MODEL_DEFAULT):
    """Explicitly require an API key for real-model calls.

    Model names starting with "claude" use Anthropic (ANTHROPIC_API_KEY);
    everything else uses OpenAI (OPENAI_API_KEY).
    """
    if not live:
        return DemoToolCallingModel()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass  # .env support is optional; environment variables still work
    if model_name.startswith("claude"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is missing. Set it in the environment or .env.")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=0)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Create .env (see .env.example).")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name, temperature=0)


def describe_messages(messages: Sequence[Any]) -> None:
    for msg in messages:
        print(f"[{msg.__class__.__name__}] {content_text(msg.content)[:500]}")
        for call in getattr(msg, "tool_calls", []) or []:
            print(f"  TOOL CALL: {call.get('name')}({call.get('args')})")


def _sample_input():
    from langchain_core.messages import HumanMessage
    return {"messages": [HumanMessage(content="Sell 25 Apple shares and purchase 10 Microsoft stocks")]}


def build_graph(llm: Any, tools: Sequence[Any], *, variant: str = "react"):
    """Build four progressively more capable LangGraph lesson examples."""
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import ToolException
    from langgraph.graph import START, END, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    def tool_error_message(error: Exception) -> str:
        if not isinstance(error, (ToolException, ValueError)):
            raise error
        return str(error)

    if variant not in ("basic", "manual-tools", "conditional", "prebuilt", "react"):
        raise ValueError(f"Unknown graph variant: {variant}")
    bound = llm.bind_tools(tools)
    builder = StateGraph(MessagesState)

    def call_model(state: MessagesState):
        return {"messages": [bound.invoke(state["messages"])]}

    builder.add_node("agent", call_model)
    builder.add_edge(START, "agent")
    if variant == "basic":
        builder.add_edge("agent", END)
        return builder.compile()

    if variant == "manual-tools" or variant == "conditional":
        mapping = {t.name: t for t in tools}

        def execute_tools(state: MessagesState):
            last = state["messages"][-1]
            results = []
            for call in getattr(last, "tool_calls", []) or []:
                try:
                    results.append(mapping[call["name"]].invoke(call))
                except (ToolException, ValueError) as exc:
                    results.append(ToolMessage(content=tool_error_message(exc), name=call["name"],
                                               tool_call_id=call["id"], status="error"))
            return {"messages": results}

        builder.add_node("tools", execute_tools)
    else:
        builder.add_node("tools", ToolNode(tools, handle_tool_errors=tool_error_message))

    if variant == "manual-tools":
        builder.add_edge("agent", "tools")
    else:
        builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent" if variant == "react" else END)
    return builder.compile()


def _run_tools(llm: Any) -> None:
    tools = make_tools()
    print("Stock price tool type:", type(tools[0]))
    print("Stock price example (NOT live):", tools[0].invoke({"name": "Apple"}))
    print("Simulated trade:", tools[1].invoke({"name": "Microsoft", "qty": 10}))


def _run_tool_calling(llm: Any) -> None:
    tools = make_tools()
    bound = llm.bind_tools(tools)
    questions = (
        "Sell 25 Apple shares and purchase 10 Microsoft stocks",
        "Explain the P/E ratio in one line",
    )
    for question in questions:
        print("\nQUESTION:", question)
        result = bound.invoke(question)
        describe_messages([result])


def _run_manual_execution(llm: Any) -> None:
    tools = make_tools()
    tool_map = {t.name: t for t in tools}
    response = llm.bind_tools(tools).invoke("Sell 25 Apple shares and buy 10 Microsoft stocks")
    describe_messages([response])
    executed = [tool_map[call["name"]].invoke(call) for call in response.tool_calls]
    print("Executed tool responses (ToolMessage objects):")
    describe_messages(executed)
    no_tools = llm.bind_tools(tools).invoke("Explain the P/E ratio in one line")
    print("No-call case:")
    describe_messages([no_tools])


def _run_graph(llm: Any, variant: str, graph_mermaid: bool = False) -> None:
    graph = build_graph(llm, make_tools(), variant=variant)
    if graph_mermaid:
        print(graph.get_graph().draw_mermaid())
    result = graph.invoke(_sample_input(), config={"recursion_limit": 8})
    describe_messages(result["messages"])


def _run_tavily(llm: Any, live: bool) -> None:
    if not live:
        print("Tavily is skipped in offline mode. Enable --live and set TAVILY_API_KEY + OPENAI_API_KEY.")
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is missing; see .env.example")
    
    TavilySearch = import_module("langchain_tavily").TavilySearch
    search = TavilySearch(max_results=3)
    response = llm.bind_tools([search]).invoke("Find public information about IBM as a company")
    describe_messages([response])
    print("Tool calls are only proposed here; no search executed until invoked.")


STOCK_TOOL_NAMES = ("stock_price", "buy_stock", "sell_stock")
NEWS_TOOL_NAMES = ("market_news", "market_sentiment")


def server_tool_name(name: str, server: str) -> str:
    """Strip the server prefix MCPAdapter adds when several servers are connected
    (stock_stock_price -> stock_price). With a single server, names are unprefixed."""
    prefix = server + "_"
    rest = name[len(prefix):] if name.startswith(prefix) else None
    return rest if rest in STOCK_TOOL_NAMES + NEWS_TOOL_NAMES else name


def mcp_connections(*, include_news: bool = False) -> dict[str, Any]:
    """Modern MCPAdapter config, using paths independent of notebook CWD."""
    servers: dict[str, dict[str, Any]] = {
        "stock": {
            "command": sys.executable,
            "args": [str(ROOT / "stock_market_mcp_server.py")],
        }
    }
    if include_news:
        servers["news"] = {
            "url": "http://127.0.0.1:8000/mcp",
        }
    return {"mcpServers": servers}


async def list_mcp_tools(adapter, *, include_news: bool) -> list[Any]:
    tools = await adapter.list_tools()
    if include_news:
        available = {server_tool_name(tool.name, "news") for tool in tools}
        if not set(NEWS_TOOL_NAMES).issubset(available):
            raise RuntimeError("News MCP tools are unavailable. Start python news_mcp_server.py before using --include-news.")
    return tools


async def _run_mcp_client(*, include_news: bool) -> list[Any]:
    # LangChain >=1.4 has native MCP support. The legacy separate package
    # langchain-mcp-adapters 0.3.2 requires MCP SDK <2 and conflicts with v2.
    from langchain.mcp import MCPAdapter
    async with MCPAdapter(mcp_connections(include_news=include_news)) as adapter:
        tools = await list_mcp_tools(adapter, include_news=include_news)
        print("All MCP tools:", [t.name for t in tools])
        stock_tools = [t for t in tools if server_tool_name(t.name, "stock") in STOCK_TOOL_NAMES]
        print("STDIO stock server tools:", [t.name for t in stock_tools])
        if include_news:
            news_tools = [t for t in tools if server_tool_name(t.name, "news") in NEWS_TOOL_NAMES]
            print("HTTP news server tools:", [t.name for t in news_tools])
        example_price_tool = next((t for t in stock_tools if server_tool_name(t.name, "stock") == "stock_price"), None)
        if example_price_tool is not None:
            result = await example_price_tool.ainvoke({"name": "Apple"})
            print("STDIO sample tool result:", content_text(result), "(simulated, not market data)")
        return tools


async def _run_mcp_graph(llm: Any, *, include_news: bool, react: bool, graph_mermaid: bool):
    from langchain.mcp import MCPAdapter
    from langchain_core.messages import HumanMessage
    async with MCPAdapter(mcp_connections(include_news=include_news)) as adapter:
        tools = await list_mcp_tools(adapter, include_news=include_news)
        graph = build_graph(llm, tools, variant="react" if react else "prebuilt")
        if graph_mermaid:
            print(graph.get_graph().draw_mermaid())
        if react:
            question = "What is the price of Google stocks?"
        else:
            question = "What is the price of Apple stocks?"
        result = await graph.ainvoke({"messages": [HumanMessage(content=question)]},
                                     config={"recursion_limit": 8})
        describe_messages(result["messages"])
        return result


async def run_topic(
    topic: str = "all", *, live: bool = False, model_name: str = MODEL_DEFAULT,
    include_news: bool = False, graph_mermaid: bool = False,
) -> None:
    """Notebook-friendly coroutine: use `await run_topic('tools')` in Jupyter."""
    if topic not in (*TOPICS, "all"):
        raise ValueError(f"Unknown topic {topic!r}; choose from {TOPICS}")
    llm = choose_llm(live=live, model_name=model_name)
    topics = TOPICS if topic == "all" else (topic,)
    for name in topics:
        print("\n" + "=" * 14 + f" {name.upper()} " + "=" * 14)
        if name == "tools":
            _run_tools(llm)
        elif name == "tool-calling":
            _run_tool_calling(llm)
        elif name == "manual-execution":
            _run_manual_execution(llm)
        elif name in ("graph-basic", "graph-manual-tools", "graph-conditional", "graph-prebuilt"):
            variant = name.removeprefix("graph-")
            _run_graph(llm, variant=variant, graph_mermaid=graph_mermaid)
        elif name == "tavily":
            _run_tavily(llm, live)
        elif name == "mcp-client":
            await _run_mcp_client(include_news=include_news)
        elif name == "mcp-graph":
            await _run_mcp_graph(llm, include_news=include_news, react=False,
                                 graph_mermaid=graph_mermaid)
        elif name == "react":
            await _run_mcp_graph(llm, include_news=include_news, react=True,
                                 graph_mermaid=graph_mermaid)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list", action="store_true", help="List all chapter topics")
    modes.add_argument("--self-test", action="store_true", help="Offline standard-library checks")
    parser.add_argument("--topic", choices=(*TOPICS, "all"), default="all")
    parser.add_argument("--live", action="store_true", help="Use a live model API (OpenAI; Anthropic for claude-* models); default is the deterministic offline demo")
    parser.add_argument("--model", default=MODEL_DEFAULT, help="Chat model ID used with --live; names starting with claude use Anthropic")
    parser.add_argument("--include-news", action="store_true",
                        help="Connect HTTP MCP server on 127.0.0.1:8000 (start separately)")
    parser.add_argument("--graph-mermaid", action="store_true", help="Print Mermaid source, no network request")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(TOPICS))
        return 0
    if args.self_test:
        self_test()
        return 0
    try:
        asyncio.run(run_topic(args.topic, live=args.live, model_name=args.model,
                              include_news=args.include_news, graph_mermaid=args.graph_mermaid))
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}. Install with: python -m pip install -r requirements.txt",
              file=sys.stderr)
        return 2
    except (RuntimeError, ConnectionError) as exc:
        print(f"Could not complete example: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
