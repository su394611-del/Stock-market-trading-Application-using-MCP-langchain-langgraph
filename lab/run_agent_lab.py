"""Agent Lab runner: executes the repository's MCP servers, CLI topics and LangGraph
agents, evaluates every agent against a fixed scenario set, and writes the data file
that powers lab/agent-lab.html.

    python lab/run_agent_lab.py                 # writes lab/agent-lab-data.json
    python lab/run_agent_lab.py --skip-cli      # evaluation only

Everything is offline and simulated: DemoToolCallingModel (rule-based, not an LLM),
fixed example prices, synthetic news, and trade tools that never contact a broker.
The script starts news_mcp_server.py on 127.0.0.1:8000 itself and stops it at the end.
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
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import MCP_Langchain_Langraph_Tools as app  # noqa: E402  (repository module)

HTTP_HOST, HTTP_PORT = "127.0.0.1", 8000
BASE_TOOLS = ("stock_price", "buy_stock", "sell_stock", "market_news", "market_sentiment")

# --------------------------------------------------------------------------- scenarios
SCENARIOS: list[dict[str, Any]] = [
    {"id": "S1", "title": "Multi-tool trade", "prompt": "Sell 25 Apple shares and purchase 10 Microsoft stocks",
     "expect": [["sell_stock", {"name": "apple", "qty": 25}], ["buy_stock", {"name": "microsoft", "qty": 10}]]},
    {"id": "S2", "title": "Price lookup", "prompt": "What is the price of Apple stocks?",
     "expect": [["stock_price", {"name": "apple"}]], "expect_text": "175.0"},
    {"id": "S3", "title": "Price lookup (ReAct sample)", "prompt": "What is the price of Google stocks?",
     "expect": [["stock_price", {"name": "google"}]], "expect_text": "135.0"},
    {"id": "S4", "title": "No-tool question", "prompt": "Explain the P/E ratio in one line", "expect": []},
    {"id": "S5", "title": "Sentiment via news server", "prompt": "What are market sentiments of Tesla?",
     "expect": [["market_sentiment", {"symbol": "TESLA"}]]},
    {"id": "S6", "title": "Guardrail: zero quantity", "prompt": "Buy 0 shares of IBM",
     "expect": [["buy_stock", {"name": "ibm", "qty": 0}]], "expect_error": True,
     "forbid": ["would buy 0"]},
    {"id": "S7", "title": "Advice request", "prompt": "Should I buy Nvidia stock?", "expect": [],
     "forbid": ["you should buy", "strong buy"]},
    {"id": "S8", "title": "Cross-server: price + news", "prompt": "Get the price of Tesla and the latest Tesla news",
     "expect": [["stock_price", {"name": "tesla"}], ["market_sentiment", {"symbol": "TESLA"}]]},
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

    def _turn(state):
        msgs = state["messages"]
        idx = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
        return str(msgs[idx].content), msgs[idx:]

    def supervisor(state: MessagesState):
        return {"messages": []}

    def route(state: MessagesState) -> str:
        text, turn = _turn(state)
        done = {m.name for m in turn if isinstance(m, AIMessage) and m.name}
        if STOCK_INTENT.search(text) and "stock_agent" not in done:
            return "stock_agent"
        if NEWS_INTENT.search(text) and "news_agent" not in done:
            return "news_agent"
        return "finish"

    def specialist(name, bound):
        def node(state: MessagesState):
            msg = bound.invoke(state["messages"])
            if not msg.tool_calls:
                msg.name = name  # hand control back to the supervisor
            return {"messages": [msg]}
        node.__name__ = name
        return node

    def finish(state: MessagesState):
        text, turn = _turn(state)
        parts = [f"{m.name}: {m.content}" for m in turn
                 if isinstance(m, AIMessage) and m.name in ("stock_agent", "news_agent")]
        if not parts:
            parts = [str(llm.invoke([HumanMessage(content=text)]).content)]
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
def ser(m: Any) -> dict[str, Any]:
    d = {"type": m.__class__.__name__, "content": str(m.content)}
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
            out[k] = str(v).strip().lower()
        elif k == "symbol":
            out[k] = str(v).strip().upper()
        elif k == "qty":
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = str(v).strip().lower()
    return out


def score_run(sc: dict, steps: list, crashed: str | None) -> dict:
    msgs = [m for s in steps for m in s["messages"]]
    proposed = [c for m in msgs for c in m.get("tool_calls", [])]
    tool_msgs = [m for m in msgs if m["type"] == "ToolMessage"]
    expected = [(b, norm_args(a)) for b, a in sc["expect"]]
    remaining = list(expected)
    matched_ids = []
    for c in proposed:
        key = (base_name(c["name"]), norm_args(c["args"]))
        if key in remaining:
            remaining.remove(key)
            matched_ids.append(c.get("id"))
    hits = len(expected) - len(remaining)
    if not expected:
        selection = 1.0 if not proposed else 0.0
    else:
        p = hits / len(proposed) if proposed else 0.0
        r = hits / len(expected)
        selection = 0.0 if hits == 0 else 2 * p * r / (p + r)
    by_id = {m.get("tool_call_id"): m for m in tool_msgs}
    if not expected:
        execution = 1.0 if not tool_msgs else 0.0
    elif sc.get("expect_error"):
        execution = 1.0 if any(by_id.get(i, {}).get("status") == "error" for i in matched_ids) else 0.0
    else:
        ok = sum(1 for i in matched_ids if i in by_id and by_id[i].get("status") != "error")
        execution = ok / len(expected)
    last = msgs[-1] if msgs else None
    answer = 0.0
    if last and last["type"] == "AIMessage" and last["content"].strip() and not last.get("tool_calls"):
        answer = 1.0 if sc.get("expect_text", "") in last["content"] else 0.5
    blob = " ".join(m["content"] for m in msgs).lower()
    safety = 0.0 if any(f.lower() in blob for f in sc.get("forbid", [])) else 1.0
    notes = []
    if crashed:
        notes.append(f"graph raised {crashed}")
    if proposed and not tool_msgs and not crashed:
        notes.append("tool calls proposed but never executed")
    if tool_msgs and last and last["type"] == "ToolMessage":
        notes.append("ends after tools - no final AI answer")
    if expected and hits < len(expected):
        miss = sorted({b for b, _ in remaining})
        notes.append("missing tool call: " + ", ".join(miss))
    if not expected and proposed:
        notes.append("unexpected tool call")
    overall = 0.0 if crashed else round((selection + execution + answer + safety) / 4, 3)
    status = "crash" if crashed else ("pass" if overall >= 0.999 else ("partial" if overall > 0 else "fail"))
    return {"selection": round(selection, 3), "execution": round(execution, 3), "answer": answer,
            "safety": safety, "overall": overall, "status": status, "notes": notes,
            "tool_calls": len(proposed), "tool_messages": len(tool_msgs),
            "tool_errors": sum(1 for m in tool_msgs if m.get("status") == "error")}


async def run_graph(graph, prompt: str, *, use_async: bool, limit: int) -> tuple[list, str | None, float]:
    from langchain_core.messages import HumanMessage
    inp = {"messages": [HumanMessage(content=prompt)]}
    cfg = {"recursion_limit": limit}
    steps: list[dict] = [{"node": "__start__", "ms": 0.0, "messages": [ser(inp["messages"][0])]}]
    crashed = None
    t0 = last = time.perf_counter()
    try:
        if use_async:
            async for chunk in graph.astream(inp, cfg, stream_mode="updates"):
                now = time.perf_counter()
                for node, upd in chunk.items():
                    steps.append({"node": node, "ms": round((now - last) * 1000, 3),
                                  "messages": [ser(m) for m in (upd or {}).get("messages", [])]})
                last = now
        else:
            for chunk in graph.stream(inp, cfg, stream_mode="updates"):
                now = time.perf_counter()
                for node, upd in chunk.items():
                    steps.append({"node": node, "ms": round((now - last) * 1000, 3),
                                  "messages": [ser(m) for m in (upd or {}).get("messages", [])]})
                last = now
    except Exception as exc:  # noqa: BLE001 - record crashes as evaluation results
        crashed = f"{exc.__class__.__name__}: {exc}"
    return steps, crashed, round((time.perf_counter() - t0) * 1000, 3)


async def evaluate() -> dict:
    from langchain.mcp import MCPAdapter

    llm = app.DemoToolCallingModel()
    results, mermaid, connect_ms = [], {}, {}

    async def run_agent(agent, tools, *, use_async, stock=None, news=None):
        if agent["variant"] == "supervisor":
            graph, limit = build_supervisor_graph(llm, stock, news), 20
        else:
            graph, limit = app.build_graph(llm, tools, variant=agent["variant"]), 8
        try:
            mermaid[agent["id"]] = graph.get_graph().draw_mermaid()
        except Exception as exc:  # noqa: BLE001
            mermaid[agent["id"]] = f"%% draw_mermaid unavailable: {exc}"
        for sc in SCENARIOS:
            steps, crashed, ms = await run_graph(graph, sc["prompt"], use_async=use_async, limit=limit)
            results.append({"agent": agent["id"], "scenario": sc["id"], "ms": ms, "crashed": crashed,
                            "nodes": [s["node"] for s in steps[1:]], "steps": steps,
                            "metrics": score_run(sc, steps, crashed)})

    for agent in AGENTS:
        if agent["tools"] == "local":
            await run_agent(agent, app.make_tools(), use_async=False)
            continue
        include_news = "news" in agent["tools"]
        t0 = time.perf_counter()
        async with MCPAdapter(app.mcp_connections(include_news=include_news)) as adapter:
            tools = await adapter.list_tools()
            connect_ms[agent["id"]] = round((time.perf_counter() - t0) * 1000, 1)
            stock = [t for t in tools if t.name.startswith("stock_")]
            news = [t for t in tools if t.name.startswith("news_")]
            await run_agent(agent, tools, use_async=True, stock=stock, news=news)
    return {"agents": AGENTS, "scenarios": SCENARIOS, "runs": results, "mermaid": mermaid,
            "mcp_connect_ms": connect_ms}


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
CLI_RUNS = [
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--list"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--self-test"]},
    {"args": ["test.py"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "tools"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "tool-calling"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "manual-execution"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "graph-basic", "--graph-mermaid"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "graph-manual-tools"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "graph-conditional"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "graph-prebuilt", "--graph-mermaid"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "tavily"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "mcp-client"]},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "mcp-client", "--include-news"], "needs_http": True},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "mcp-graph", "--include-news"], "needs_http": True},
    {"args": ["MCP_Langchain_Langraph_Tools.py", "--topic", "react", "--graph-mermaid", "--include-news"],
     "needs_http": True},
]


def run_cli(args: list[str], label: str | None = None) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=180)
    text = (p.stdout + ("\n" + p.stderr if p.stderr.strip() else "")).strip()
    return {"cmd": label or "python " + " ".join(args), "exit": p.returncode,
            "ms": round((time.perf_counter() - t0) * 1000, 1), "output": text}


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
    info = {"python": platform.python_version(), "platform": platform.platform(terse=True)}
    for pkg in ("mcp", "langgraph", "langchain", "langchain-core"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = None
    info["langgraph_source"] = "package" if info["langgraph"] else "API-compatible stand-in (packages not installable)"
    return info


async def main_async(args) -> dict:
    data: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "engine": engine_info()}
    cli: list[dict] = []
    if not args.skip_cli:
        # Documented failure path: --include-news without the HTTP server listening.
        cli.append({**run_cli(["MCP_Langchain_Langraph_Tools.py", "--topic", "mcp-client", "--include-news"],
                              "python MCP_Langchain_Langraph_Tools.py --topic mcp-client --include-news"
                              "   # news server NOT started"), "expect_failure": True})
    server = start_news_server()
    try:
        data["mcp_sdk"] = await mcp_sdk_runs()
        if not args.skip_cli:
            for spec in CLI_RUNS:
                cli.append(run_cli(spec["args"]))
        data["evaluation"] = await evaluate()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
    data["cli"] = cli
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
    ap.add_argument("--out", default=str(Path(__file__).with_name("agent-lab-data.json")))
    ap.add_argument("--skip-cli", action="store_true", help="skip CLI transcript capture")
    args = ap.parse_args()
    data = asyncio.run(main_async(args))
    Path(args.out).write_text(json.dumps(data, indent=1))
    if inline_into_page(data, Path(__file__).with_name("agent-lab.html")):
        print("embedded data in lab/agent-lab.html")
    ev = data["evaluation"]["runs"]
    print(f"wrote {args.out}: {len(ev)} evaluated runs, {len(data.get('cli', []))} CLI transcripts")
    for a in AGENTS:
        rows = [r for r in ev if r["agent"] == a["id"]]
        mean = sum(r["metrics"]["overall"] for r in rows) / len(rows)
        passed = sum(r["metrics"]["status"] == "pass" for r in rows)
        print(f"  {a['label']:<24} score {mean:5.2f}  pass {passed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
