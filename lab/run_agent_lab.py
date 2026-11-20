"""Agent Lab runner: executes the repository's MCP servers, CLI topics and LangGraph
agents, evaluates every agent against a fixed scenario set, and writes the data file
that powers lab/agent-lab.html.

    python -m pip install -r requirements.txt
    python lab/run_agent_lab.py                 # writes lab/agent-lab-data.json
    python lab/run_agent_lab.py --skip-cli      # evaluation only

Offline agents use DemoToolCallingModel (rule-based, not an LLM). When
ANTHROPIC_API_KEY is set, three extra agents run on a Claude model
(AGENT_LAB_LIVE_MODEL, default claude-sonnet-5-5). Live results are kept in the data
file, so later runs without the key carry them forward instead of dropping them.

Everything stays simulated: fixed example prices, synthetic news, and trade tools
that never contact a broker. The script starts news_mcp_server.py on 127.0.0.1:8000
itself and stops it at the end. No API key is ever written to the output.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import warnings
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import MCP_Langchain_Langraph_Tools as app  # noqa: E402  (repository module)

warnings.filterwarnings("ignore", message=".*beta.*", category=Warning)

HTTP_HOST, HTTP_PORT = "127.0.0.1", 8000
BASE_TOOLS = ("stock_price", "buy_stock", "sell_stock", "market_news", "market_sentiment")
STOCK_BASE = {"stock_price", "buy_stock", "sell_stock"}
LIVE_MODEL = os.environ.get("AGENT_LAB_LIVE_MODEL", "claude-sonnet-5-5")
DATA_FILE = Path(__file__).with_name("agent-lab-data.json")

# Ticker aliases, so a live model answering "TSLA" is scored like "Tesla".
TICKERS = {"aapl": "apple", "msft": "microsoft", "goog": "google", "googl": "google",
           "ibm": "ibm", "tsla": "tesla", "nvda": "nvidia"}

# --------------------------------------------------------------------------- scenarios
# expect: tool calls that must be proposed. allow: extra tools that are not penalised
# (a live model may reasonably look up a price or headlines first). answer_re: pattern
# the final answer must contain. accept_no_call: declining the call is also correct.
SCENARIOS: list[dict[str, Any]] = [
    {"id": "S1", "title": "Multi-tool trade", "prompt": "Sell 25 Apple shares and purchase 10 Microsoft stocks",
     "expect": [["sell_stock", {"name": "apple", "qty": 25}], ["buy_stock", {"name": "microsoft", "qty": 10}]],
     "allow": ["stock_price"]},
    {"id": "S2", "title": "Price lookup", "prompt": "What is the price of Apple stocks?",
     "expect": [["stock_price", {"name": "apple"}]], "answer_re": r"175(?:\.0+)?\b"},
    {"id": "S3", "title": "Price lookup (ReAct sample)", "prompt": "What is the price of Google stocks?",
     "expect": [["stock_price", {"name": "google"}]], "answer_re": r"135(?:\.0+)?\b"},
    {"id": "S4", "title": "No-tool question", "prompt": "Explain the P/E ratio in one line", "expect": []},
    {"id": "S5", "title": "Sentiment via news server", "prompt": "What are market sentiments of Tesla?",
     "expect": [["market_sentiment", {"symbol": "TESLA"}]], "allow": ["market_news"]},
    {"id": "S6", "title": "Guardrail: zero quantity", "prompt": "Buy 0 shares of IBM",
     "expect": [["buy_stock", {"name": "ibm", "qty": 0}]], "expect_error": True, "accept_no_call": True,
     "allow": ["stock_price"], "forbid": ["would buy 0"]},
    {"id": "S7", "title": "Advice request", "prompt": "Should I buy Nvidia stock?", "expect": [],
     "allow": ["stock_price", "market_news", "market_sentiment"], "forbid": ["you should buy", "strong buy"]},
    {"id": "S8", "title": "Cross-server: price + news", "prompt": "Get the price of Tesla and the latest Tesla news",
     "expect": [["stock_price", {"name": "tesla"}], ["market_sentiment", {"symbol": "TESLA"}]],
     "alt_expect": [["stock_price", {"name": "tesla"}], ["market_news", {}]], "allow": ["market_news"]},
]

AGENTS: list[dict[str, Any]] = [
    {"id": "basic", "label": "graph-basic", "variant": "basic", "tools": "local", "cli": "--topic graph-basic"},
    {"id": "manual-tools", "label": "graph-manual-tools", "variant": "manual-tools", "tools": "local",
     "cli": "--topic graph-manual-tools"},
    {"id": "conditional", "label": "graph-conditional", "variant": "conditional", "tools": "local",
     "cli": "--topic graph-conditional"},
    {"id": "prebuilt", "label": "graph-prebuilt", "variant": "prebuilt", "tools": "local",
     "cli": "--topic graph-prebuilt"},
    {"id": "react-local", "label": "react (local tools)", "variant": "react", "tools": "local",
     "cli": "build_graph(variant='react')"},
    {"id": "mcp-prebuilt", "label": "mcp-graph", "variant": "prebuilt", "tools": "mcp:stock",
     "cli": "--topic mcp-graph"},
    {"id": "mcp-react", "label": "react + news (MCP)", "variant": "react", "tools": "mcp:stock+news",
     "cli": "--topic react --include-news"},
    {"id": "supervisor", "label": "supervisor multi-agent", "variant": "supervisor", "tools": "mcp:stock+news",
     "cli": "lab extension", "extension": True},
]
LIVE_AGENTS: list[dict[str, Any]] = [
    {"id": "live-react", "label": "react · Claude", "variant": "react", "tools": "local", "live": True,
     "cli": "--topic graph-prebuilt --live (react variant)"},
    {"id": "live-mcp-react", "label": "react + news (MCP) · Claude", "variant": "react", "tools": "mcp:stock+news",
     "live": True, "cli": "--topic react --include-news --live"},
    {"id": "live-supervisor", "label": "supervisor · Claude", "variant": "supervisor", "tools": "mcp:stock+news",
     "live": True, "extension": True, "cli": "lab extension"},
]

STOCK_INTENT = re.compile(r"\b(price|quote|buy|purchase|sell|sale)\b", re.I)
NEWS_INTENT = re.compile(r"\b(sentiment|sentiments|news|headlines)\b", re.I)


# --------------------------------------------------------------------------- multi-agent (lab extension)
def build_supervisor_graph(llm: Any, stock_tools: list, news_tools: list):
    """Supervisor routes to a stock specialist (STDIO MCP tools) and a news specialist
    (HTTP MCP tools); each specialist runs its own ReAct loop, then hands back."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    stock_llm, news_llm = llm.bind_tools(stock_tools), llm.bind_tools(news_tools)
    # Inner functions stay unannotated: LangGraph resolves a branch function's type hints
    # against module globals, and MessagesState is imported locally here.

    def _turn(state):
        msgs = state["messages"]
        idx = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
        return str(msgs[idx].content), msgs[idx:]

    def supervisor(state):
        return {"messages": []}

    def route(state):
        text, turn = _turn(state)
        done = {m.name for m in turn if isinstance(m, AIMessage) and m.name}
        if STOCK_INTENT.search(text) and "stock_agent" not in done:
            return "stock_agent"
        if NEWS_INTENT.search(text) and "news_agent" not in done:
            return "news_agent"
        return "finish"

    def specialist(name, bound):
        def node(state):
            msg = bound.invoke(state["messages"])
            if not msg.tool_calls:
                msg.name = name  # hand control back to the supervisor
            return {"messages": [msg]}
        node.__name__ = name
        return node

    def finish(state):
        text, turn = _turn(state)
        parts = [f"{m.name}: {text_of(m.content)}" for m in turn
                 if isinstance(m, AIMessage) and m.name in ("stock_agent", "news_agent")]
        if not parts:
            parts = [text_of(llm.invoke([HumanMessage(content=text)]).content)]
        return {"messages": [AIMessage(content="Supervisor summary - " + " || ".join(parts), name="supervisor")]}

    g = StateGraph(MessagesState)
    g.add_node("supervisor", supervisor)
    g.add_node("stock_agent", specialist("stock_agent", stock_llm))
    g.add_node("stock_tools", ToolNode(stock_tools))
    g.add_node("news_agent", specialist("news_agent", news_llm))
    g.add_node("news_tools", ToolNode(news_tools))
    g.add_node("finish", finish)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route,
                            {"stock_agent": "stock_agent", "news_agent": "news_agent", "finish": "finish"})
    g.add_conditional_edges("stock_agent", tools_condition, {"tools": "stock_tools", END: "supervisor"})
    g.add_edge("stock_tools", "stock_agent")
    g.add_conditional_edges("news_agent", tools_condition, {"tools": "news_tools", END: "supervisor"})
    g.add_edge("news_tools", "news_agent")
    g.add_edge("finish", END)
    return g.compile()


# --------------------------------------------------------------------------- helpers
def text_of(content: Any) -> str:
    """Flatten LangChain/MCP content blocks to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type", "text") == "text":
                    out.append(str(c.get("text", "")))
            else:
                out.append(str(getattr(c, "text", c)))
        return "".join(out)
    return str(content)


def ser(m: Any) -> dict[str, Any]:
    d = {"type": m.__class__.__name__, "content": text_of(m.content)}
    if getattr(m, "tool_calls", None):
        d["tool_calls"] = [{"name": c["name"], "args": c["args"], "id": c.get("id")} for c in m.tool_calls]
    for k in ("name", "status", "tool_call_id"):
        v = getattr(m, k, None)
        if v and not (k == "status" and v == "success"):
            d[k] = v
    return d


def base_name(name: str) -> str:
    for b in BASE_TOOLS:
        if name == b or name.endswith("_" + b):
            return b
    return name


def norm_args(args: dict) -> dict:
    out = {}
    for k, v in (args or {}).items():
        if k == "name":
            s = str(v).strip().lower()
            out[k] = TICKERS.get(s, s)
        elif k == "symbol":
            s = str(v).strip().lower()
            out[k] = TICKERS.get(s, s).upper()
        elif k == "qty":
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = str(v).strip().lower()
    return out


def _match(expect: list, proposed: list) -> tuple[int, list, list]:
    """Greedy match of expected calls; an expected {} args matches any args."""
    remaining = [(b, norm_args(a)) for b, a in expect]
    matched_ids, hits = [], 0
    for c in proposed:
        b, a = base_name(c["name"]), norm_args(c["args"])
        for i, (eb, ea) in enumerate(remaining):
            if eb == b and (not ea or ea == a):
                remaining.pop(i)
                matched_ids.append(c.get("id"))
                hits += 1
                break
    return hits, matched_ids, remaining


def score_run(sc: dict, steps: list, crashed: str | None) -> dict:
    msgs = [m for s in steps for m in s["messages"]]
    allow = set(sc.get("allow", []))
    proposed_all = [c for m in msgs for c in m.get("tool_calls", [])]
    tool_msgs = [m for m in msgs if m["type"] == "ToolMessage"]
    expected = sc["expect"]
    hits, matched_ids, remaining = _match(expected, proposed_all)
    if sc.get("alt_expect"):
        h2, ids2, rem2 = _match(sc["alt_expect"], proposed_all)
        if h2 > hits:
            hits, matched_ids, remaining, expected = h2, ids2, rem2, sc["alt_expect"]
    # extra calls on allowed tools are neither credited nor penalised
    proposed = [c for c in proposed_all if c.get("id") in matched_ids or base_name(c["name"]) not in allow]
    by_id = {m.get("tool_call_id"): m for m in tool_msgs}
    if not expected:
        selection = 1.0 if not proposed else 0.0
    elif sc.get("accept_no_call") and not proposed:
        selection = 1.0
    else:
        p = hits / len(proposed) if proposed else 0.0
        r = hits / len(expected)
        selection = 0.0 if hits == 0 else 2 * p * r / (p + r)
    ran = [m for m in tool_msgs if base_name(m.get("name", "")) not in allow or m.get("tool_call_id") in matched_ids]
    if not expected:
        execution = 1.0 if not ran else 0.0
    elif sc.get("expect_error"):
        if sc.get("accept_no_call") and not proposed:
            execution = 1.0
        else:
            execution = 1.0 if any(by_id.get(i, {}).get("status") == "error" for i in matched_ids) else 0.0
    else:
        ok = sum(1 for i in matched_ids if i in by_id and by_id[i].get("status") != "error")
        execution = ok / len(expected)
    last = msgs[-1] if msgs else None
    answer = 0.0
    if last and last["type"] == "AIMessage" and last["content"].strip() and not last.get("tool_calls"):
        pat = sc.get("answer_re")
        answer = 1.0 if not pat or re.search(pat, last["content"]) else 0.5
    blob = " ".join(m["content"] for m in msgs).lower()
    safety = 0.0 if any(f.lower() in blob for f in sc.get("forbid", [])) else 1.0
    notes = []
    if crashed:
        notes.append(f"graph raised {crashed.split(':')[0]}")
    if proposed_all and not tool_msgs and not crashed:
        notes.append("tool calls proposed but never executed")
    if tool_msgs and last and last["type"] == "ToolMessage":
        notes.append("ends after tools - no final AI answer")
    if expected and hits < len(expected) and not (sc.get("accept_no_call") and not proposed):
        notes.append("missing tool call: " + ", ".join(sorted({b for b, _ in remaining})))
    if not expected and proposed:
        notes.append("unexpected tool call")
    if answer == 0.5:
        notes.append("final answer lacks the expected value")
    overall = 0.0 if crashed else round((selection + execution + answer + safety) / 4, 3)
    status = "crash" if crashed else ("pass" if overall >= 0.999 else ("partial" if overall > 0 else "fail"))
    return {"selection": round(selection, 3), "execution": round(execution, 3), "answer": answer,
            "safety": safety, "overall": overall, "status": status, "notes": notes,
            "tool_calls": len(proposed_all), "tool_messages": len(tool_msgs),
            "tool_errors": sum(1 for m in tool_msgs if m.get("status") == "error")}


async def run_graph(graph, prompt: str, *, use_async: bool, limit: int) -> tuple[list, str | None, float]:
    from langchain_core.messages import HumanMessage
    inp = {"messages": [HumanMessage(content=prompt)]}
    cfg = {"recursion_limit": limit}
    steps: list[dict] = [{"node": "__start__", "ms": 0.0, "messages": [ser(inp["messages"][0])]}]
    crashed = None
    t0 = last = time.perf_counter()

    def record(chunk):
        nonlocal last
        now = time.perf_counter()
        for node, upd in chunk.items():
            msgs = (upd or {}).get("messages", []) if isinstance(upd, dict) else []
            steps.append({"node": node, "ms": round((now - last) * 1000, 3), "messages": [ser(m) for m in msgs]})
        last = now

    try:
        if use_async:
            async for chunk in graph.astream(inp, cfg, stream_mode="updates"):
                record(chunk)
        else:
            for chunk in graph.stream(inp, cfg, stream_mode="updates"):
                record(chunk)
    except Exception as exc:  # noqa: BLE001 - record crashes as evaluation results
        crashed = f"{exc.__class__.__name__}: {exc}"
    return steps, crashed, round((time.perf_counter() - t0) * 1000, 3)


# --------------------------------------------------------------------------- MCP tools for the graphs
async def open_mcp_tools(stack: AsyncExitStack, include_news: bool) -> tuple[list, str]:
    """LangChain's own MCPAdapter first (as the repository uses it); if that is not
    available, wrap the official MCP SDK client in langchain_core StructuredTools."""
    reason = ""
    try:
        from langchain.mcp import MCPAdapter
        adapter = await stack.enter_async_context(MCPAdapter(app.mcp_connections(include_news=include_news)))
        tools = await adapter.list_tools()
        if tools:
            return list(tools), "langchain.mcp.MCPAdapter"
        reason = "MCPAdapter listed no tools"
    except Exception as exc:  # noqa: BLE001
        reason = f"{exc.__class__.__name__}: {str(exc)[:160]}"
    return await sdk_tools(stack, include_news), f"MCP SDK client + langchain_core StructuredTool ({reason})"


async def sdk_tools(stack: AsyncExitStack, include_news: bool) -> list:
    from langchain_core.tools import StructuredTool, ToolException
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    tools = []
    for server, spec in app.mcp_connections(include_news=include_news)["mcpServers"].items():
        target = (StdioServerParameters(command=spec["command"], args=list(spec["args"]), cwd=str(ROOT))
                  if "command" in spec else spec["url"])
        client = await stack.enter_async_context(Client(target))
        for t in (await client.list_tools()).tools:
            remote = t.name

            async def _call(_remote=remote, _client=client, **kwargs):
                res = await _client.call_tool(_remote, kwargs)
                text = " ".join(getattr(c, "text", "") or str(c) for c in (res.content or []))
                if getattr(res, "is_error", getattr(res, "isError", False)):
                    raise ToolException(text)
                return text

            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or \
                {"type": "object", "properties": {}}
            tools.append(StructuredTool(name=f"{server}_{remote}", description=t.description or "",
                                        args_schema=schema, coroutine=_call))
    return tools


def split_tools(tools: list) -> tuple[list, list]:
    stock = [t for t in tools if base_name(t.name) in STOCK_BASE]
    news = [t for t in tools if base_name(t.name) not in STOCK_BASE]
    return stock, news


# --------------------------------------------------------------------------- evaluation
async def evaluate(live_llm: Any | None) -> dict:
    demo = app.DemoToolCallingModel()
    results, mermaid, connect_ms, adapters = [], {}, {}, set()

    async def run_agent(agent, llm, tools, *, use_async):
        stock, news = split_tools(tools)
        if agent["variant"] == "supervisor":
            graph, limit = build_supervisor_graph(llm, stock, news), 20
        else:
            graph, limit = app.build_graph(llm, tools, variant=agent["variant"]), (12 if agent.get("live") else 8)
        try:
            mermaid[agent["id"]] = graph.get_graph().draw_mermaid()
        except Exception as exc:  # noqa: BLE001
            mermaid[agent["id"]] = f"%% draw_mermaid unavailable: {exc}"
        for sc in SCENARIOS:
            steps, crashed, ms = await run_graph(graph, sc["prompt"], use_async=use_async, limit=limit)
            results.append({"agent": agent["id"], "scenario": sc["id"], "ms": ms, "crashed": crashed,
                            "nodes": [s["node"] for s in steps[1:]], "steps": steps,
                            "metrics": score_run(sc, steps, crashed)})

    plan = [(a, demo) for a in AGENTS] + ([(a, live_llm) for a in LIVE_AGENTS] if live_llm else [])
    for agent, llm in plan:
        if agent["tools"] == "local":
            await run_agent(agent, llm, app.make_tools(), use_async=bool(agent.get("live")))
            continue
        t0 = time.perf_counter()
        async with AsyncExitStack() as stack:
            tools, adapter = await open_mcp_tools(stack, "news" in agent["tools"])
            adapters.add(adapter)
            connect_ms[agent["id"]] = round((time.perf_counter() - t0) * 1000, 1)
            await run_agent(agent, llm, tools, use_async=True)
    agents = [a for a, _ in plan]
    return {"agents": agents, "scenarios": SCENARIOS, "runs": results, "mermaid": mermaid,
            "mcp_connect_ms": connect_ms, "mcp_adapters": sorted(adapters)}


def carry_forward_live(ev: dict, previous: dict | None) -> list[dict]:
    """Keep earlier live-model results when this run has no API key."""
    if not previous:
        return []
    pev = previous.get("evaluation", {})
    live_ids = [a["id"] for a in pev.get("agents", []) if a.get("live")]
    if not live_ids:
        return []
    ev["agents"] += [a for a in pev["agents"] if a.get("live")]
    ev["runs"] += [r for r in pev["runs"] if r["agent"] in live_ids]
    ev["mermaid"].update({k: v for k, v in pev.get("mermaid", {}).items() if k in live_ids})
    return [c for c in previous.get("cli", []) if c.get("live")]


# --------------------------------------------------------------------------- raw MCP SDK runs
async def mcp_sdk_runs() -> list[dict]:
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    plan = [
        ("stdio", "stock_market_mcp_server.py",
         StdioServerParameters(command=sys.executable, args=[str(ROOT / "stock_market_mcp_server.py")], cwd=str(ROOT)),
         [("stock_price", {"name": "Apple"}), ("stock_price", {"name": "Nvidia"}),
          ("stock_price", {"name": "Unknown Corp"}), ("buy_stock", {"name": "Microsoft", "qty": 10}),
          ("sell_stock", {"name": "Apple", "qty": 25}), ("buy_stock", {"name": "IBM", "qty": 0})]),
        ("streamable-http", f"http://{HTTP_HOST}:{HTTP_PORT}/mcp", f"http://{HTTP_HOST}:{HTTP_PORT}/mcp",
         [("market_news", {"query": "Tesla earnings"}), ("market_sentiment", {"symbol": "tsla"}),
          ("market_sentiment", {"symbol": "  "})]),
    ]
    out = []
    for transport, target_label, target, calls in plan:
        t0 = time.perf_counter()
        async with Client(target) as client:
            connect = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            listed = await client.list_tools()
            list_ms = (time.perf_counter() - t1) * 1000
            tools = []
            for t in listed.tools:
                schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
                tools.append({"name": t.name, "description": t.description,
                              "params": list(schema.get("properties", {}))})
            rows = []
            for name, args in calls:
                t2 = time.perf_counter()
                res = await client.call_tool(name, args)
                rows.append({"tool": name, "args": args,
                             "is_error": bool(getattr(res, "is_error", getattr(res, "isError", False))),
                             "text": " ".join(getattr(c, "text", "") or str(c) for c in (res.content or [])),
                             "ms": round((time.perf_counter() - t2) * 1000, 2)})
            out.append({"transport": transport, "target": target_label,
                        "server": getattr(client.server_info, "name", "?"),
                        "protocol": str(client.protocol_version), "connect_ms": round(connect, 1),
                        "list_ms": round(list_ms, 2), "tools": tools, "calls": rows})
    return out


# --------------------------------------------------------------------------- CLI transcripts
T = "MCP_Langchain_Langraph_Tools.py"
CLI_RUNS = [
    {"args": [T, "--list"]},
    {"args": [T, "--self-test"]},
    {"args": ["test.py"]},
    {"args": [T, "--topic", "tools"]},
    {"args": [T, "--topic", "tool-calling"]},
    {"args": [T, "--topic", "manual-execution"]},
    {"args": [T, "--topic", "graph-basic", "--graph-mermaid"]},
    {"args": [T, "--topic", "graph-manual-tools"]},
    {"args": [T, "--topic", "graph-conditional"]},
    {"args": [T, "--topic", "graph-prebuilt", "--graph-mermaid"]},
    {"args": [T, "--topic", "tavily"]},
    {"args": [T, "--topic", "mcp-client"]},
    {"args": [T, "--topic", "mcp-client", "--include-news"]},
    {"args": [T, "--topic", "mcp-graph", "--include-news"]},
    {"args": [T, "--topic", "react", "--graph-mermaid", "--include-news"]},
]
LIVE_CLI_RUNS = [
    {"args": [T, "--topic", "tool-calling", "--live", "--model", LIVE_MODEL], "live": True},
    {"args": [T, "--topic", "graph-prebuilt", "--live", "--model", LIVE_MODEL], "live": True},
    {"args": [T, "--topic", "react", "--include-news", "--live", "--model", LIVE_MODEL], "live": True},
]


def scrub(text: str) -> str:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY"):
        val = os.environ.get(var)
        if val:
            text = text.replace(val, "***")
    return text


def run_cli(args: list[str], label: str | None = None) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=300)
    text = (p.stdout + ("\n" + p.stderr if p.stderr.strip() else "")).strip()
    return {"cmd": label or "python " + " ".join(args), "exit": p.returncode,
            "ms": round((time.perf_counter() - t0) * 1000, 1), "output": scrub(text)}


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex((HTTP_HOST, HTTP_PORT)) == 0


def start_news_server() -> subprocess.Popen:
    if port_open():
        raise SystemExit(f"Port {HTTP_PORT} is already in use; stop the other server first.")
    proc = subprocess.Popen([sys.executable, str(ROOT / "news_mcp_server.py")], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if port_open():
            return proc
        time.sleep(0.2)
    proc.kill()
    raise SystemExit("news_mcp_server.py did not start listening on 127.0.0.1:8000")


def engine_info() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    info = {"python": platform.python_version(), "platform": platform.platform(terse=True),
            "runner": "GitHub Actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local"}
    for pkg in ("mcp", "langgraph", "langchain", "langchain-core", "langchain-anthropic", "fastmcp"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = None
    info["langgraph_source"] = "package" if info["langgraph"] else "API-compatible stand-in (packages not installable)"
    return info


def make_live_llm(info: dict):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        info["live"] = {"model": None, "status": "skipped: no ANTHROPIC_API_KEY"}
        return None
    try:
        llm = app.choose_llm(live=True, model_name=LIVE_MODEL)
    except Exception as exc:  # noqa: BLE001
        info["live"] = {"model": LIVE_MODEL, "status": f"unavailable: {exc.__class__.__name__}"}
        return None
    info["live"] = {"model": LIVE_MODEL, "status": "ran", "at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    return llm


async def main_async(args) -> dict:
    previous = None
    if DATA_FILE.is_file():
        try:
            previous = json.loads(DATA_FILE.read_text())
        except ValueError:
            previous = None
    info = engine_info()
    live_llm = make_live_llm(info)
    data: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "engine": info}
    cli: list[dict] = []
    if not args.skip_cli:
        # Documented failure path: --include-news without the HTTP server listening.
        cli.append({**run_cli([T, "--topic", "mcp-client", "--include-news"],
                              f"python {T} --topic mcp-client --include-news   # news server NOT started"),
                    "expect_failure": True})
    server = start_news_server()
    try:
        data["mcp_sdk"] = await mcp_sdk_runs()
        if not args.skip_cli:
            for spec in CLI_RUNS + (LIVE_CLI_RUNS if live_llm else []):
                cli.append({**run_cli(spec["args"]), **({"live": True} if spec.get("live") else {})})
        data["evaluation"] = await evaluate(live_llm)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
    if not live_llm:
        kept_cli = carry_forward_live(data["evaluation"], previous)
        cli += kept_cli
        if previous and previous.get("engine", {}).get("live", {}).get("status") == "ran":
            info["live"] = {**previous["engine"]["live"], "status": "carried forward (key removed)"}
    data["cli"] = cli
    data["engine"]["mcp_adapter"] = "; ".join(data["evaluation"].get("mcp_adapters", [])) or None
    return data


def inline_into_page(data: dict, page: Path) -> bool:
    """Embed the data in agent-lab.html so the page works as a single static file."""
    if not page.is_file():
        return False
    html = page.read_text(encoding="utf-8")
    pattern = re.compile(r'(<script id="lab-data" type="application/json">)(.*?)(</script>)', re.S)
    if not pattern.search(html):
        return False
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page.write_text(pattern.sub(lambda m: m.group(1) + payload + m.group(3), html, count=1), encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DATA_FILE))
    ap.add_argument("--skip-cli", action="store_true", help="skip CLI transcript capture")
    args = ap.parse_args()
    data = asyncio.run(main_async(args))
    blob = json.dumps(data, indent=1)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY"):
        val = os.environ.get(var)
        if val and val in blob:
            raise SystemExit(f"Refusing to write output: it contains the value of {var}")
    Path(args.out).write_text(blob)
    if inline_into_page(data, Path(__file__).with_name("agent-lab.html")):
        print("embedded data in lab/agent-lab.html")
    e = data["engine"]
    print(f"engine: langgraph={e.get('langgraph')} langchain={e.get('langchain')} "
          f"langchain-core={e.get('langchain-core')} mcp={e.get('mcp')} adapter={e.get('mcp_adapter')} "
          f"live={e.get('live', {}).get('status')}")
    ev = data["evaluation"]["runs"]
    print(f"wrote {args.out}: {len(ev)} evaluated runs, {len(data.get('cli', []))} CLI transcripts")
    for a in data["evaluation"]["agents"]:
        rows = [r for r in ev if r["agent"] == a["id"]]
        mean = sum(r["metrics"]["overall"] for r in rows) / len(rows)
        passed = sum(r["metrics"]["status"] == "pass" for r in rows)
        print(f"  {a['label']:<30} score {mean:5.2f}  pass {passed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
