/*ENGINE-START*/
/* In-browser port of the repository's offline logic: interpret_demo_request,
   DemoToolCallingModel, the simulated tools, build_graph variants and the lab's
   supervisor graph, with langgraph 1.2 ToolNode semantics. It reproduces the recorded
   Python traces exactly (checked live below each run). Rule-based, not an LLM. */
const LabEngine = (() => {
  const PRICES = {apple:175.0, microsoft:320.0, google:135.0, alphabet:135.0, ibm:145.0, tesla:215.0, nvidia:125.0};
  const pyFloat = x => Number.isInteger(x) ? x.toFixed(1) : String(x);
  const title = s => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
  const pyRepr = s => (s.includes("'") && !s.includes('"')) ? `"${s}"` : `'${s.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
  class PyError extends Error { constructor(type, msg) { super(msg); this.pyType = type; } repr() { return `${this.pyType}(${pyRepr(this.message)})`; } }
  const word = k => new RegExp(`\\b${k}\\b`, "i");

  function fakePrice(name) { name = String(name).trim().toLowerCase(); if (!name) throw new PyError("ValueError", "name must not be empty"); return PRICES[name] ?? 100.0; }
  function simulatedTrade(action, name, qty) {
    if (action !== "buy" && action !== "sell") throw new PyError("ValueError", "action must be buy or sell");
    if (!String(name).trim() || typeof qty !== "number" || !Number.isInteger(qty) || qty < 1) throw new PyError("ValueError", "name must be supplied and qty must be a positive integer");
    return `SIMULATION ONLY: would ${action} ${qty} shares of ${name}. No trade was placed.`;
  }
  function marketNews(query) { if (!String(query).trim()) throw new PyError("ValueError", "query must not be blank"); return `SYNTHETIC DEMO: Example news search for ${pyRepr(String(query))}. No live articles were fetched.`; }
  function marketSentiment(symbol) { if (!String(symbol).trim()) throw new PyError("ValueError", "symbol must not be blank"); return `SYNTHETIC DEMO: ${String(symbol).toUpperCase()} sentiment unavailable without real market data.`; }

  function interpret(text, allowedTools) {
    const allowed = new Set(allowedTools);
    const resolve = b => allowed.has(b) ? b : ([...allowed].sort().find(n => n.endsWith("_" + b)) ?? null);
    const calls = [];
    const re = /\b(buy|purchase|sell|sale)\s+(\d+)\s+(?:(?:shares?|stocks?)\s+(?:of\s+)?)?([A-Za-z][A-Za-z.-]*)/gi;
    for (const m of text.matchAll(re)) {
      const tool = /^(sell|sale)$/i.test(m[1]) ? "sell_stock" : "buy_stock";
      const actual = resolve(tool);
      if (actual) calls.push({name: actual, args: {name: m[3], qty: parseInt(m[2], 10)}, id: `demo_call_${calls.length + 1}`});
    }
    if (calls.length) return calls;
    const priceTool = resolve("stock_price");
    if (/\b(price|quote)\b/i.test(text) && priceTool) {
      for (const c of Object.keys(PRICES)) if (word(c).test(text)) { calls.push({name: priceTool, args: {name: title(c)}, id: "demo_call_1"}); break; }
    }
    if (/\b(sentiment|sentiments|news|headlines)\b/i.test(text)) {
      const st = resolve("market_sentiment"), nt = resolve("market_news");
      if (st) { const co = Object.keys(PRICES).find(k => word(k).test(text)) ?? "market"; calls.push({name: st, args: {symbol: co.toUpperCase()}, id: `demo_call_${calls.length + 1}`}); }
      else if (nt) calls.push({name: nt, args: {query: text}, id: `demo_call_${calls.length + 1}`});
    }
    return calls;
  }

  const AI = (content, tool_calls) => tool_calls && tool_calls.length ? {type: "AIMessage", content, tool_calls} : {type: "AIMessage", content};
  function demoModel(toolNames) {
    return msgs => {
      const last = msgs[msgs.length - 1];
      const human = [...msgs].reverse().find(m => m.type === "HumanMessage");
      const text = human ? String(human.content) : "";
      if (last && last.type === "ToolMessage") {
        let start = msgs.length - 1;
        while (start > 0 && msgs[start - 1].type === "ToolMessage") start--;
        const results = msgs.slice(start);
        const failed = results.some(message => message.status === "error");
        const label = failed ? "Demo tool execution reported an error: " : "Demo tool execution complete: ";
        return AI(label + results.map(message => message.content).join(" | "));
      }
      const t = text.toLowerCase();
      if (t.includes("p/e") || t.includes("pe ratio")) return AI("P/E ratio = stock price divided by earnings per share (demo).");
      if (t.includes("should i") || t.includes("recommendation")) return AI("Demo only: no real-time analysis or investment recommendation is provided.");
      const calls = interpret(text, toolNames);
      const priorCalls = msgs.reduce((count, message) => count + (message.tool_calls?.length || 0), 0);
      calls.forEach((call, index) => { call.id = `demo_call_${priorCalls + index + 1}`; });
      return calls.length ? AI("", calls) : AI("Demo mode: no tool call selected for this question.");
    };
  }

  /* Expected validation failures become error ToolMessages for local and MCP tools. */
  const BASE = {stock_price: a => pyFloat(fakePrice(a.name)), buy_stock: a => simulatedTrade("buy", a.name, a.qty), sell_stock: a => simulatedTrade("sell", a.name, a.qty), market_news: a => marketNews(a.query), market_sentiment: a => marketSentiment(a.symbol)};
  function toolset(kind) {
    const make = (name, base, mcp) => ({name, invoke(call) {
      try { return {type: "ToolMessage", content: BASE[base](call.args || {}), name, tool_call_id: call.id}; }
      catch (e) {
        if (!(e instanceof PyError)) throw e;
        const content = mcp ? `Error executing tool ${base}: ${e.message}` : e.message;
        return {type: "ToolMessage", content, name, status: "error", tool_call_id: call.id};
      }
    }});
    if (kind === "local") return ["stock_price", "buy_stock", "sell_stock"].map(b => make(b, b, false));
    if (kind === "mcp:stock") return ["stock_price", "buy_stock", "sell_stock"].map(b => make(b, b, true));   // one server: no prefix
    const stock = ["stock_price", "buy_stock", "sell_stock"].map(b => make("stock_" + b, b, true));
    const news = ["market_news", "market_sentiment"].map(b => make("news_" + b, b, true));
    return [...stock, ...news];
  }
  const toolNode = tools => state => {
    const byName = Object.fromEntries(tools.map(t => [t.name, t]));
    const out = [];
    for (const call of state.messages[state.messages.length - 1].tool_calls || []) {
      const t = byName[call.name];
      if (!t) { out.push({type: "ToolMessage", content: `Error: ${call.name} is not a valid tool, try one of [${Object.keys(byName).join(", ")}].`, name: call.name, status: "error", tool_call_id: call.id}); continue; }
      out.push(t.invoke(call));
    }
    return out;
  };
  const manualExec = tools => state => {
    const byName = Object.fromEntries(tools.map(t => [t.name, t]));
    return (state.messages[state.messages.length - 1].tool_calls || []).map(c => byName[c.name].invoke(c));
  };
  const toolsCondition = state => { const l = state.messages[state.messages.length - 1]; return l && l.type === "AIMessage" && l.tool_calls && l.tool_calls.length ? "tools" : "__end__"; };

  function buildGraph(variant, toolKind) {
    const tools = toolset(toolKind);
    if (variant === "supervisor") return buildSupervisor(tools);
    const model = demoModel(tools.map(t => t.name));
    const nodes = {agent: s => [model(s.messages)]};
    const edges = {__start__: ["agent"]}, cond = {};
    if (variant === "basic") edges.agent = ["__end__"];
    else {
      nodes.tools = (variant === "manual-tools" || variant === "conditional") ? manualExec(tools) : toolNode(tools);
      if (variant === "manual-tools") edges.agent = ["tools"]; else cond.agent = toolsCondition;
      edges.tools = [variant === "react" ? "agent" : "__end__"];
    }
    return {nodes, edges, cond, limit: 8};
  }
  const STOCK_INTENT = /\b(price|quote|buy|purchase|sell|sale)\b/i, NEWS_INTENT = /\b(sentiment|sentiments|news|headlines)\b/i;
  function buildSupervisor(tools) {
    const stock = tools.filter(t => t.name.startsWith("stock_")), news = tools.filter(t => t.name.startsWith("news_"));
    const sm = demoModel(stock.map(t => t.name)), nm = demoModel(news.map(t => t.name)), plain = demoModel([]);
    const turn = msgs => { let i = msgs.length - 1; while (i > 0 && msgs[i].type !== "HumanMessage") i--; return [String(msgs[i].content), msgs.slice(i)]; };
    const specialist = (name, model) => s => { const m = model(s.messages); if (!m.tool_calls) m.name = name; return [m]; };
    const nodes = {
      supervisor: () => [], stock_agent: specialist("stock_agent", sm), stock_tools: toolNode(stock),
      news_agent: specialist("news_agent", nm), news_tools: toolNode(news),
      finish: s => { const [text, t] = turn(s.messages);
        let parts = t.filter(m => m.type === "AIMessage" && (m.name === "stock_agent" || m.name === "news_agent")).map(m => `${m.name}: ${m.content}`);
        if (!parts.length) parts = [plain([{type: "HumanMessage", content: text}]).content];
        return [{type: "AIMessage", content: "Supervisor summary - " + parts.join(" || "), name: "supervisor"}]; }
    };
    const route = s => { const [text, t] = turn(s.messages); const done = new Set(t.filter(m => m.type === "AIMessage" && m.name).map(m => m.name));
      if (STOCK_INTENT.test(text) && !done.has("stock_agent")) return "stock_agent";
      if (NEWS_INTENT.test(text) && !done.has("news_agent")) return "news_agent";
      return "finish"; };
    return {nodes, limit: 20, edges: {__start__: ["supervisor"], stock_tools: ["stock_agent"], news_tools: ["news_agent"], finish: ["__end__"]},
      cond: {supervisor: route, stock_agent: s => toolsCondition(s) === "tools" ? "stock_tools" : "supervisor", news_agent: s => toolsCondition(s) === "tools" ? "news_tools" : "supervisor"}};
  }

  function run(variant, toolKind, prompt) {
    const g = buildGraph(variant, toolKind);
    const human = {type: "HumanMessage", content: prompt};
    const state = {messages: [human]};
    const steps = [{node: "__start__", messages: [human]}];
    let frontier = [...g.edges.__start__], n = 0, crashed = null;
    try {
      while (frontier.length) {
        const node = frontier.shift();
        if (node === "__end__") continue;
        if (++n > g.limit) throw new PyError("GraphRecursionError", `Recursion limit of ${g.limit} reached without hitting a stop condition.`);
        const added = g.nodes[node](state);
        state.messages.push(...added);
        steps.push({node, messages: added.map(m => ({...m}))});
        frontier.push(...(g.cond[node] ? [g.cond[node](state)] : (g.edges[node] || ["__end__"])));
      }
    } catch (e) { if (!(e instanceof PyError)) throw e; crashed = `${e.pyType}: ${e.message}`; }
    return {steps, crashed, nodes: steps.slice(1).map(s => s.node)};
  }
  return {run, interpret};
})();
/*ENGINE-END*/

(() => {
"use strict";
const DATA = JSON.parse(document.getElementById("lab-data").textContent || "{}");
const EV = DATA.evaluation || {agents: [], scenarios: [], runs: [], mermaid: {}};
const $ = s => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v); } for (const k of kids) if (k != null) e.append(k); return e; };
const SVGNS = "http://www.w3.org/2000/svg";
const sv = (tag, attrs = {}) => { const e = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = (x, d = 2) => Number(x).toFixed(d);
const runOf = (a, s) => EV.runs.find(r => r.agent === a && r.scenario === s);
const agentOf = id => EV.agents.find(a => a.id === id);
const median = xs => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

/* ---------------- notes from the README / wiki, with what the runs showed ---------------- */
const AGENT_NOTES = {
  "basic": {src: "Wiki · Graph Variants", doc: "START → agent → END. The model may propose tool calls but nothing executes them.", obs: "Confirmed: every tool-needing scenario ends on an AIMessage with pending tool calls, so execution and answer score 0."},
  "manual-tools": {src: "Wiki · Graph Variants", doc: "Always agent → tools → END with the custom execute_tools node; no conditional shortcut.", obs: "The custom executor records the invalid quantity as a ToolMessage with error status."},
  "conditional": {src: "Wiki · Graph Variants", doc: "tools_condition routes to the custom execute_tools node, or ends when there are no calls.", obs: "Records the validation error on S6. Skips the tools node when no calls are proposed (S4, S5, S7)."},
  "prebuilt": {src: "Wiki · Graph Variants", doc: "tools_condition + LangGraph ToolNode; run the tools, then end.", obs: "Ends on a ToolMessage. On S6 that message carries the invalid quantity error."},
  "react-local": {src: "Wiki · Graph Variants", doc: "ReAct: loop tools → agent until the model stops calling tools.", obs: "Returns a final message after tool calls, including a validation error on S6. This graph has the local stock tools."},
  "mcp-prebuilt": {src: "README · Start the MCP servers", doc: "STDIO: the client launches stock_market_mcp_server.py as a child process — do not start it manually.", obs: "Confirmed: MCPAdapter spawned the server itself. With a single server, tool names are not prefixed (stock_price). Prebuilt routing still ends on a ToolMessage."},
  "mcp-react": {src: "README · Start the MCP servers", doc: "Start news_mcp_server.py in another terminal; --include-news fails if the HTTP server is not listening.", obs: "With both servers running, the agent discovers five tools. With the HTTP server down, the CLI exits with startup instructions."},
  "supervisor": {src: "lab/run_agent_lab.py", doc: "A supervisor routes the request to a stock specialist (STDIO MCP tools) and a news specialist (HTTP MCP tools). Each runs its own ReAct loop and hands back, then a finish node summarises.", obs: "Passes 8/8. It takes up to 10 node executions (S8), against 3 for single-agent ReAct, in exchange for scoped tool access per agent."}
};
const CLI_NOTES = [
  [/--list/, "README · Run commands", "Prints the 11 topic names."],
  [/--self-test/, "Wiki · Testing & Validation", "Standard-library smoke test: prices, simulated trades, regex parsing, file presence."],
  [/test\.py/, "Wiki · Testing & Validation", "Unit tests cover tool parsing, validation, MCP prefixes and graph error handling."],
  [/--topic tools$/, "README · Topics", "Tool creation and direct invocation."],
  [/tool-calling/, "README · Topics", "LLM tool calls: several calls in one message, and a no-tool case."],
  [/manual-execution/, "README · Topics", "Manual tool execution producing ToolMessage objects."],
  [/graph-basic/, "Wiki · Graph Variants", "--graph-mermaid prints Mermaid text generated from the compiled graph, not an image."],
  [/graph-manual-tools/, "Wiki · Graph Variants", "Unconditional agent → tools → END."],
  [/graph-conditional/, "Wiki · Graph Variants", "tools_condition with the custom executor."],
  [/graph-prebuilt/, "Wiki · Graph Variants", "ToolNode + tools_condition."],
  [/tavily/, "Wiki · Offline vs Live", "Skipped offline: requires --live with OPENAI_API_KEY and TAVILY_API_KEY."],
  [/NOT started/, "README · Start the MCP servers", "Confirms that --include-news exits with instructions when the HTTP server is unavailable."],
  [/mcp-client --include-news$/, "Wiki · MCP Integration", "Terminal A runs news_mcp_server.py; terminal B runs the client. Five tools across both transports, prefixed by server name."],
  [/mcp-client$/, "README · Start the MCP servers", "STDIO only: the adapter spawns the stock server as a child process. With one server the tool names carry no prefix."],
  [/mcp-graph/, "Wiki · Execution Sequence", "MCP-backed graph: the adapter discovers tools, the graph calls stock_stock_price over STDIO."],
  [/react/, "Wiki · Execution Sequence", "ReAct loop: tools → agent until no calls; recursion_limit=8 bounds the loop."]
];
const FLOW = [
  ["A human prompt is placed in MessagesState.", "Every trace starts with the HumanMessage at __start__."],
  ["The bound offline demo model (or live model) returns an AIMessage.", "agent / specialist nodes append one AIMessage per visit."],
  ["tools_condition chooses tool execution if calls are present.", "Dashed edges in the graph view are the conditional routes."],
  ["Tool execution produces ToolMessage objects.", "Local validation failures and MCP tool errors arrive as error ToolMessages carrying the reason."],
  ["Prebuilt graphs end after the tools; the ReAct graph returns to the agent.", "prebuilt: 2 node executions; react: 3 (see Run graphs)."],
  ["The CLI prints messages and tool-call metadata; no order is submitted.", "All trade results read “SIMULATION ONLY … No trade was placed.”"]
];

/* ---------------- header badges ---------------- */
function badges() {
  const e = DATA.engine || {};
  const b = $("#badges");
  const add = (cls, html, title) => b.append(el("span", {class: "badge " + cls, html, title: title || ""}));
  add("real", `MCP SDK <b>${esc(e.mcp || "?")}</b> · real servers`, "Both MCP servers ran and were called through the official MCP Python SDK client");
  if (e.langgraph) add("real", `LangGraph <b>${esc(e.langgraph)}</b>`, "langgraph package used for every graph");
  else add("stand", `LangGraph <b>API stand-in</b>`, "langgraph/langchain were not installable where this data was generated; the repository code ran against a minimal API-compatible stand-in");
  if (e.langchain || e["langchain-core"]) add("real", `LangChain <b>${esc(e.langchain || "")}</b>${e["langchain-core"] ? ` · core <b>${esc(e["langchain-core"])}</b>` : ""}`, e.mcp_adapter ? "MCP tools via " + e.mcp_adapter : "");
  add("", `Python <b>${esc(e.python || "?")}</b>${e.runner === "GitHub Actions" ? " · GitHub Actions" : ""}`);
  const live = e.live || {};
  if (live.model && /ran|carried/.test(live.status || "")) add("real", `live · <b>${esc(live.model)}</b>`, `Live agents ran on ${live.model}${live.at ? " at " + live.at : ""}; recorded results`);
  add("", `offline · DemoToolCallingModel`);
  const adapter = e.mcp_adapter ? ` MCP tools reached the graphs through ${esc(e.mcp_adapter)}.` : "";
  $("#foot").innerHTML = `Data generated ${esc(DATA.generated_at || "")}${e.runner === "GitHub Actions" ? " on GitHub Actions" : ""} by <code>python lab/run_agent_lab.py</code> with ${e.langgraph ? `langgraph ${esc(e.langgraph)}` : "a LangGraph stand-in"}.${adapter} Simulation only: fixed example prices, synthetic news, trade tools that never contact a broker; the offline model is rule-based, not an LLM. Not investment advice.`;
}

/* ---------------- tabs ---------------- */
function tabs() {
  const list = [...document.querySelectorAll(".tab")];
  const show = t => { list.forEach(x => { const on = x === t; x.setAttribute("aria-selected", on); x.tabIndex = on ? 0 : -1; document.getElementById(x.getAttribute("aria-controls")).hidden = !on; }); };
  list.forEach((t, i) => { t.addEventListener("click", () => show(t)); t.addEventListener("keydown", e => { if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = list[(i + (e.key === "ArrowRight" ? 1 : list.length - 1)) % list.length]; n.focus(); show(n); } }); });
  return show;
}

/* ---------------- tooltip ---------------- */
const tip = $("#tip");
function showTip(html, x, y) { tip.innerHTML = html; tip.style.opacity = 1; const w = tip.offsetWidth, h = tip.offsetHeight; tip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + "px"; tip.style.top = (y + h + 20 > window.innerHeight ? y - h - 12 : y + 14) + "px"; }
function hideTip() { tip.style.opacity = 0; }
function bindTip(node, html) { node.addEventListener("mousemove", e => showTip(typeof html === "function" ? html() : html, e.clientX, e.clientY)); node.addEventListener("mouseleave", hideTip); node.addEventListener("focus", () => { const r = node.getBoundingClientRect(); showTip(typeof html === "function" ? html() : html, r.left, r.bottom); }); node.addEventListener("blur", hideTip); }

/* ---------------- graph drawing ---------------- */
const LAYOUT = {
  basic: {agent: [280, 170], __end__: [280, 310]},
  "manual-tools": {agent: [280, 128], tools: [280, 232], __end__: [280, 338]},
  conditional: {__start__: [190, 32], agent: [190, 145], tools: [400, 245], __end__: [190, 345]},
  prebuilt: {__start__: [190, 32], agent: [190, 145], tools: [400, 245], __end__: [190, 345]},
  react: {__start__: [175, 32], agent: [175, 175], tools: [410, 175], __end__: [175, 335]},
  supervisor: {supervisor: [280, 100], stock_agent: [100, 205], news_agent: [460, 205], finish: [280, 245], stock_tools: [100, 335], news_tools: [460, 335], __end__: [280, 352]}
};
const SUBTITLE = {agent: "bound.invoke(messages)", supervisor: "route(state)", stock_agent: "ReAct · stock tools", news_agent: "ReAct · news tools", finish: "summary node", stock_tools: "ToolNode · STDIO MCP", news_tools: "ToolNode · HTTP MCP"};
function parseEdges(mm) {
  const out = [];
  for (const line of String(mm || "").split("\n")) {
    const m = line.trim().match(/^(\w+)\s+(-->|-\.->|-\.\s.*?\s\.->|--\s.*?\s-->)\s+(\w+);?$/);
    if (m && !out.some(e => e.from === m[1] && e.to === m[3])) out.push({from: m[1], to: m[3], cond: m[2].startsWith("-.")});
  }
  return out;
}
function nodeBox(name) { return name.startsWith("__") ? [110, 34] : [150, 50]; }
function clip(cx, cy, w, h, tx, ty) { const dx = tx - cx, dy = ty - cy; if (!dx && !dy) return [cx, cy]; const s = Math.min(Math.abs((w / 2 + 3) / (dx || 1e-9)), Math.abs((h / 2 + 3) / (dy || 1e-9))); return [cx + dx * s, cy + dy * s]; }
let G = {nodes: {}, edges: {}};
function drawGraph(agent) {
  const svg = $("#graph"); svg.innerHTML = "";
  const layout = {__start__: [280, 32], ...LAYOUT[agent.variant]};
  const edges = parseEdges(EV.mermaid[agent.id]);
  const defs = sv("defs");
  [["arr", "#4d6a8c"], ["arrhot", "#7be2f0"]].forEach(([id, c]) => { const m = sv("marker", {id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"}); m.append(sv("path", {d: "M0,0 L10,5 L0,10 z", fill: c})); defs.append(m); });
  svg.append(defs);
  G = {nodes: {}, edges: {}};
  const eg = sv("g"), ng = sv("g"); svg.append(eg, ng);
  const has = (a, b) => edges.some(e => e.from === a && e.to === b);
  for (const e of edges) {
    const A = layout[e.from], B = layout[e.to]; if (!A || !B) continue;
    const [wa, ha] = nodeBox(e.from), [wb, hb] = nodeBox(e.to);
    let d;
    if (has(e.to, e.from)) {
      const mx = (A[0] + B[0]) / 2, my = (A[1] + B[1]) / 2, dx = B[0] - A[0], dy = B[1] - A[1], len = Math.hypot(dx, dy) || 1;
      const off = 30, cx = mx + (-dy / len) * off, cy = my + (dx / len) * off;
      const p1 = clip(A[0], A[1], wa, ha, cx, cy), p2 = clip(B[0], B[1], wb, hb, cx, cy);
      d = `M${p1[0]},${p1[1]} Q${cx},${cy} ${p2[0]},${p2[1]}`;
    } else {
      const p1 = clip(A[0], A[1], wa, ha, B[0], B[1]), p2 = clip(B[0], B[1], wb, hb, A[0], A[1]);
      d = `M${p1[0]},${p1[1]} L${p2[0]},${p2[1]}`;
    }
    const g = sv("g", {class: "edge" + (e.cond ? " cond" : "")});
    const p = sv("path", {d, "marker-end": "url(#arr)"}); g.append(p); eg.append(g);
    G.edges[e.from + ">" + e.to] = g;
  }
  for (const [name, [x, y]] of Object.entries(layout)) {
    const [w, h] = nodeBox(name), term = name.startsWith("__");
    const g = sv("g", {class: "node" + (term ? " term" : "")});
    g.append(sv("rect", {x: x - w / 2, y: y - h / 2, width: w, height: h, rx: term ? 17 : 12}));
    const t = sv("text", {x, y: term ? y + 4 : y - 2}); t.textContent = name; g.append(t);
    if (!term) { const s = sv("text", {x, y: y + 14, class: "st"}); s.textContent = (name === "tools" ? (agent.tools.startsWith("mcp") ? "ToolNode · MCP tools" : (["manual-tools", "conditional"].includes(agent.variant) ? "custom execute_tools" : "ToolNode")) : SUBTITLE[name] || ""); g.append(s); }
    ng.append(g); G.nodes[name] = g;
  }
}
function paint(seq, upto, crashedAt) {
  Object.values(G.nodes).forEach(n => n.classList.remove("active", "visited", "crash"));
  Object.values(G.edges).forEach(e => { e.classList.remove("hot", "walked"); e.querySelector("path").setAttribute("marker-end", "url(#arr)"); });
  for (let i = 0; i <= upto && i < seq.length; i++) {
    const n = G.nodes[seq[i]]; if (n) n.classList.add(i === upto ? "active" : "visited");
    if (i > 0) { const e = G.edges[seq[i - 1] + ">" + seq[i]]; if (e) { e.classList.add(i === upto ? "hot" : "walked"); if (i === upto) e.querySelector("path").setAttribute("marker-end", "url(#arrhot)"); } }
  }
  if (crashedAt != null && upto >= crashedAt) { const n = G.nodes[seq[crashedAt]]; if (n) { n.classList.remove("active"); n.classList.add("crash"); } }
}

/* ---------------- runner ---------------- */
const R = {agent: EV.agents.find(a => a.id === "supervisor") ? "supervisor" : (EV.agents[0] || {}).id, scenario: EV.scenarios.find(s => s.id === "S8") ? "S8" : "S1", run: null, i: -1, timer: null};
function msgView(m, node) {
  const kind = m.type === "HumanMessage" ? "human" : m.type === "ToolMessage" ? (m.status === "error" ? "tool err" : "tool") : "ai";
  const head = el("div", {class: "who"}, el("span", {text: m.type + (m.name ? ` · ${m.name}` : "") + (m.status === "error" ? " · error" : "")}), el("span", {text: node}));
  const body = el("div", {class: "body", text: m.content || (m.tool_calls ? "" : "∅")});
  const box = el("div", {class: "msg " + kind}, head, body);
  for (const c of m.tool_calls || []) box.append(el("div", {class: "call", text: `TOOL CALL ${c.name}(${JSON.stringify(c.args)})`}));
  return box;
}
function currentPreset() { const sc = EV.scenarios.find(s => s.id === R.scenario); return sc && $("#prompt").value.trim() === sc.prompt ? sc : null; }
function prepare() {
  stop();
  const agent = agentOf(R.agent), sc = currentPreset(), prompt = $("#prompt").value.trim();
  const rec = sc ? runOf(agent.id, sc.id) : null;
  if (agent.live && !rec) {
    R.run = null; R.seq = ["__start__"]; R.i = -1; paint(R.seq, -1);
    $("#log").innerHTML = ""; $("#log").append(el("div", {class: "empty", text: `${agent.label} runs a live model, so only its recorded runs can be replayed here. Pick a scenario, or choose an offline agent to run your own prompt in the browser.`}));
    $("#scorebar").innerHTML = ""; $("#parity").textContent = ""; $("#source").innerHTML = "Source: <b>recorded live runs only</b>";
    $("#step-info").textContent = "No recorded run for this prompt."; $("#step-ms").textContent = "";
    return false;
  }
  const live = agent.live ? null : LabEngine.run(agent.variant, agent.tools, prompt);
  R.run = rec ? {steps: rec.steps, crashed: rec.crashed, nodes: rec.nodes, rec, live} : {...live, rec: null, live};
  R.seq = ["__start__", ...R.run.nodes];
  R.crashAt = null;
  if (R.run.crashed) { R.crashAt = R.seq.length; R.seq.push(crashNode(agent, R.run)); } else R.seq.push("__end__");
  R.i = -1;
  $("#log").innerHTML = ""; $("#scorebar").innerHTML = ""; $("#parity").textContent = ""; $("#parity").className = "parity";
  $("#source").innerHTML = rec ? `Source: <b>recorded ${agent.live ? "live-model" : "Python"} run</b> (${esc(rec.scenario)} · ${fmt(rec.ms, rec.ms < 1 ? 3 : 1)} ms)` : `Source: <b>in-browser engine</b> (custom prompt)`;
  paint(R.seq, -1);
  $("#step-info").textContent = "Ready: " + (R.seq.length - 1) + " transitions.";
  $("#step-ms").textContent = "";
  return true;
}
function crashNode(agent, run) {   // node that raised = the one routing entered next
  const last = run.nodes[run.nodes.length - 1] || "agent";
  const step = run.steps[run.steps.length - 1], msg = step && step.messages[step.messages.length - 1];
  if (msg && msg.tool_calls && msg.tool_calls.length) return {stock_agent: "stock_tools", news_agent: "news_tools"}[last] || "tools";
  return last;
}
function step() {
  if (!R.run && !prepare()) return false;
  if (R.i >= R.seq.length - 1) { stop(); return false; }
  R.i++;
  paint(R.seq, R.i, R.crashAt);
  const s = R.run.steps[R.i];
  const node = R.seq[R.i];
  if (s && R.i < R.run.steps.length) { if (R.i === 0) $("#log").innerHTML = ""; s.messages.forEach(m => $("#log").append(msgView(m, node))); if (!s.messages.length && node !== "__start__") $("#log").append(el("div", {class: "empty", text: `${node}: no messages added (routing only)`})); $("#log").scrollTop = 1e6; }
  if (R.crashAt != null && R.i === R.crashAt) $("#log").append(el("div", {class: "msg err"}, el("div", {class: "who"}, el("span", {text: "Graph raised"}), el("span", {text: node})), el("div", {class: "body", text: R.run.crashed})));
  $("#step-info").innerHTML = `Step <b>${R.i}/${R.seq.length - 1}</b> · <b>${esc(node)}</b>`;
  $("#step-ms").textContent = s && s.ms != null && R.i > 0 ? `${fmt(s.ms, 3)} ms` : "";
  if (R.i === R.seq.length - 1) finishRun();
  return true;
}
function finishRun() {
  stop();
  const rec = R.run.rec, sb = $("#scorebar");
  if (rec) {
    const m = rec.metrics;
    sb.append(statusEl(m.status), ...[["selection", m.selection], ["execution", m.execution], ["answer", m.answer], ["safety", m.safety], ["overall", m.overall]].map(([k, v]) => el("span", {class: "pill", html: `${k} <b>${fmt(v)}</b>`})));
    m.notes.forEach(n => sb.append(el("span", {class: "pill", text: n})));
    const p = $("#parity");
    if (!R.run.live) { p.className = "parity"; p.textContent = `Recorded live run on ${((DATA.engine || {}).live || {}).model || "a live model"}; the in-browser engine reproduces only the offline demo model.`; }
    else { const same = JSON.stringify(strip(R.run.live)) === JSON.stringify(strip(rec)); p.className = "parity " + (same ? "ok" : "bad");
      p.textContent = same ? "✓ The in-browser engine reproduces this recorded trace exactly." : "✕ The in-browser engine differs from the recorded trace."; }
  } else {
    const calls = R.run.steps.flatMap(s => s.messages).flatMap(m => m.tool_calls || []).length;
    sb.append(el("span", {class: "pill", html: `node executions <b>${R.run.nodes.length}</b>`}), el("span", {class: "pill", html: `tool calls <b>${calls}</b>`}), el("span", {class: "pill", text: R.run.crashed ? "graph raised" : "completed"}));
  }
}
function strip(r) { return {nodes: r.nodes, crashed: r.crashed, msgs: r.steps.map(s => s.messages.map(m => [m.type, m.content, (m.tool_calls || []).map(c => [c.name, c.args]), m.status || null, m.name || null]))}; }
function statusEl(s) { const icon = {pass: "✓", partial: "◐", crash: "✕", fail: "✕"}[s]; return el("span", {class: "status s-" + s, html: `<i aria-hidden="true">${icon}</i>${s}`}); }
function play() { if ((!R.run || R.i >= R.seq.length - 1) && !prepare()) return; stop(); $("#run-btn").textContent = "❚❚ Pause"; R.timer = setInterval(() => { if (!step()) stop(); }, +$("#speed").value); step(); }
function stop() { if (R.timer) clearInterval(R.timer); R.timer = null; $("#run-btn").textContent = "▶ Run"; }
function selectAgent(id, sync = true) {
  R.agent = id; const a = agentOf(id);
  document.querySelectorAll("#agent-chips .chip").forEach(c => c.setAttribute("aria-pressed", c.dataset.id === id));
  $("#graph-title").textContent = `${a.label} — LangGraph`;
  $("#graph-sub").textContent = `variant: ${a.variant} · tools: ${a.tools} · ${a.extension ? "lab extension" : "repo: " + a.cli}`;
  $("#mermaid").textContent = EV.mermaid[id] || "";
  const n = AGENT_NOTES[id] || (a.live ? {src: "Live model · " + (((DATA.engine || {}).live || {}).model || "Claude"), doc: "Same graph as the offline agent of this shape, with the rule-based DemoToolCallingModel replaced by a real tool-calling model through choose_llm(live=True). The supervisor routing stays rule-based.", obs: "Scored with the same checks; tickers such as TSLA count as Tesla, and extra price or news look-ups are not penalised."} : null);
  $("#agent-note").innerHTML = n ? `<div class="src">${esc(n.src)}</div><p data-lab-style="margin-4px-0"><strong>Docs:</strong> ${esc(n.doc)}</p><p data-lab-style="margin-4px-0"><strong>Observed:</strong> ${esc(n.obs)}</p>` : "";
  drawGraph(a); R.run = null; if (sync) prepare();
}
function selectScenario(id) { R.scenario = id; const sc = EV.scenarios.find(s => s.id === id); $("#prompt").value = sc.prompt; document.querySelectorAll("#sc-chips .chip").forEach(c => c.setAttribute("aria-pressed", c.dataset.id === id)); prepare(); }
function runner() {
  const ac = $("#agent-chips");
  EV.agents.forEach(a => { const b = el("button", {class: "chip", "data-id": a.id, type: "button", "aria-pressed": "false"}, a.label); if (a.live) b.append(el("small", {class: "ext", text: "live"})); else if (a.extension) b.append(el("small", {class: "ext", text: "ext"})); b.addEventListener("click", () => selectAgent(a.id)); ac.append(b); });
  const scc = $("#sc-chips");
  EV.scenarios.forEach(s => { const b = el("button", {class: "chip", "data-id": s.id, type: "button", "aria-pressed": "false", title: s.prompt}, s.id); b.append(el("small", {text: s.title})); b.addEventListener("click", () => selectScenario(s.id)); scc.append(b); });
  $("#run-btn").addEventListener("click", () => R.timer ? stop() : play());
  $("#step-btn").addEventListener("click", () => { stop(); if ((!R.run || R.i >= R.seq.length - 1) && !prepare()) return; step(); });
  $("#reset-btn").addEventListener("click", prepare);
  $("#prompt").addEventListener("input", () => { R.run = null; document.querySelectorAll("#sc-chips .chip").forEach(c => c.setAttribute("aria-pressed", currentPreset() && c.dataset.id === R.scenario)); $("#source").innerHTML = currentPreset() ? "Source: <b>recorded Python run</b>" : "Source: <b>in-browser engine</b> (custom prompt)"; });
  $("#prompt").addEventListener("keydown", e => { if (e.key === "Enter") play(); });
  selectAgent(R.agent, false); selectScenario(R.scenario);
}

/* ---------------- evaluation matrix ---------------- */
const RAMP = ["#122a45", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"];
function rampColor(v) { const t = Math.max(0, Math.min(1, v)) * (RAMP.length - 1), i = Math.floor(t), f = t - i; if (i >= RAMP.length - 1) return RAMP[RAMP.length - 1]; const a = hex(RAMP[i]), b = hex(RAMP[i + 1]); return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(",")})`; }
function hex(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
function matrix(show) {
  const t = $("#matrix");
  const head = el("tr", {}, el("th", {text: "Agent"}));
  EV.scenarios.forEach(s => head.append(el("th", {scope: "col", html: `${s.id}<small>${esc(s.title)}</small>`})));
  head.append(el("th", {scope: "col", "data-lab-style": "text-align-right", text: "Mean"}));
  t.append(el("thead", {}, head));
  const tb = el("tbody");
  EV.agents.forEach(a => {
    const tr = el("tr", {}, el("th", {scope: "row", html: `${esc(a.label)}${a.extension ? ' <small data-lab-style="color-var-violet">ext</small>' : ""}<br><small data-lab-style="color-var-dim-font-weight-600">${esc(a.tools)}</small>`}));
    let sum = 0;
    EV.scenarios.forEach(s => {
      const r = runOf(a.id, s.id), m = r.metrics; sum += m.overall;
      const bg = m.status === "crash" ? "#3a1820" : rampColor(m.overall);
      const dark = m.overall >= 0.7 && m.status !== "crash";
      const icon = {pass: "✓", partial: "◐", crash: "✕", fail: "✕"}[m.status];
      const btn = el("button", {type: "button", "aria-label": `${a.label}, ${s.id} ${s.title}: ${m.status}, score ${fmt(m.overall)}. Replay run.`, html: `<span class="v">${fmt(m.overall)}</span><span class="st">${icon} ${m.status}</span>`});
      btn.style.color = dark ? "#07121f" : "#edf5ff";
      const td = el("td", {class: "cell"}, btn); td.style.background = bg; if (m.status === "crash") td.style.boxShadow = "inset 0 0 0 2px #d03b3b";
      bindTip(btn, `<b>${esc(a.label)} · ${s.id}</b><div class="r">${esc(s.prompt)}</div><div data-lab-style="margin-top-6px">selection <b>${fmt(m.selection)}</b> · execution <b>${fmt(m.execution)}</b><br>answer <b>${fmt(m.answer)}</b> · safety <b>${fmt(m.safety)}</b></div><div class="r" data-lab-style="margin-top-4px">${m.notes.length ? esc(m.notes.join("; ")) : "all checks met"}</div><div class="r">path: ${esc(r.nodes.join(" → ") || "—")}</div>`);
      btn.addEventListener("click", () => { hideTip(); selectAgent(a.id, false); selectScenario(s.id); show(document.getElementById("t-run")); play(); });
      tr.append(td);
    });
    tr.append(el("td", {class: "mean", text: fmt(sum / EV.scenarios.length)}));
    tb.append(tr);
  });
  t.append(tb);
  // metric table (accessible alternative)
  const mt = el("table", {class: "tbl"});
  mt.append(el("thead", {}, el("tr", {}, ...["Agent", "Scenario", "Status", "Selection", "Execution", "Answer", "Safety", "Overall", "Nodes", "Notes"].map(h => el("th", {text: h})))));
  const mb = el("tbody");
  EV.runs.forEach(r => { const m = r.metrics; mb.append(el("tr", {}, el("td", {text: agentOf(r.agent).label}), el("td", {text: r.scenario}), el("td", {text: m.status}), ...[m.selection, m.execution, m.answer, m.safety, m.overall].map(v => el("td", {class: "num", text: fmt(v)})), el("td", {class: "num", text: r.nodes.length}), el("td", {text: m.notes.join("; ")}))); });
  mt.append(mb); $("#matrix-table").append(mt);
  $("#matrix-table-btn").addEventListener("click", e => { const h = $("#matrix-table"); h.hidden = !h.hidden; e.target.textContent = h.hidden ? "Show metric table" : "Hide metric table"; });
}

/* ---------------- charts (hand-built SVG, one axis each) ---------------- */
function hbar(container, rows, {max = 1, fmtV = v => fmt(v), color = "var(--series-1)", tipFn, labelW = 150} = {}) {
  const W = 520, rowH = 26, top = 6, H = top + rows.length * rowH + 22, x0 = labelW, x1 = W - 46;
  const svg = sv("svg", {viewBox: `0 0 ${W} ${H}`, role: "img"});
  const grid = sv("g", {class: "grid"});
  for (let k = 0; k <= 4; k++) { const x = x0 + (x1 - x0) * k / 4; grid.append(sv("line", {x1: x, x2: x, y1: top, y2: H - 20})); const t = sv("text", {x, y: H - 6, "text-anchor": "middle"}); t.textContent = fmtV(max * k / 4); grid.append(t); }
  svg.append(grid);
  rows.forEach((r, i) => {
    const y = top + i * rowH, w = Math.max(r.value > 0 ? 3 : 0, (x1 - x0) * r.value / max);
    const lt = sv("text", {x: x0 - 8, y: y + 16, "text-anchor": "end"}); lt.textContent = r.label; svg.append(lt);
    const g = sv("g");
    const hit = sv("rect", {x: 0, y, width: W, height: rowH, class: "hit"});
    const bar = sv("path", {class: "mark", fill: r.color || color, d: roundedRight(x0, y + 6, w, rowH - 12, 4)});
    const vt = sv("text", {x: x0 + w + 6, y: y + 16, class: "lbl"}); vt.textContent = fmtV(r.value);
    g.append(hit, bar, vt); svg.append(g);
    if (tipFn) bindTip(hit, tipFn(r));
  });
  container.append(svg);
}
function roundedRight(x, y, w, h, r) { if (w <= 0) return ""; r = Math.min(r, w, h / 2); return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`; }
function groupedHBar(container, rows, series, {max, tipFn, labelW = 150}) {
  const W = 520, rowH = 34, top = 30, H = top + rows.length * rowH + 22, x0 = labelW, x1 = W - 40, bh = 10;
  const svg = sv("svg", {viewBox: `0 0 ${W} ${H}`, role: "img"});
  let lx = x0; series.forEach(s => { svg.append(sv("rect", {x: lx, y: 8, width: 10, height: 10, rx: 2, fill: s.color})); const t = sv("text", {x: lx + 15, y: 17}); t.textContent = s.label; svg.append(t); lx += 15 + s.label.length * 6.4 + 18; });
  const grid = sv("g", {class: "grid"});
  for (let k = 0; k <= 4; k++) { const x = x0 + (x1 - x0) * k / 4; grid.append(sv("line", {x1: x, x2: x, y1: top, y2: H - 20})); const t = sv("text", {x, y: H - 6, "text-anchor": "middle"}); t.textContent = Math.round(max * k / 4); grid.append(t); }
  svg.append(grid);
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    const lt = sv("text", {x: x0 - 8, y: y + 20, "text-anchor": "end"}); lt.textContent = r.label; svg.append(lt);
    const hit = sv("rect", {x: 0, y, width: W, height: rowH, class: "hit"}); svg.append(hit);
    series.forEach((s, j) => { const v = r[s.key], w = (x1 - x0) * v / max, yy = y + 5 + j * (bh + 2); svg.append(sv("path", {class: "mark", fill: s.color, d: roundedRight(x0, yy, Math.max(v ? 2 : 0, w), bh, 3)})); const vt = sv("text", {x: x0 + w + 5, y: yy + 9, class: "lbl", "font-size": 10}); vt.textContent = v; svg.append(vt); });
    if (tipFn) bindTip(hit, tipFn(r));
  });
  container.append(svg);
}
function dotPlot(container, rows, {max, tipFn, labelW = 150}) {
  const W = 520, rowH = 26, top = 6, H = top + rows.length * rowH + 22, x0 = labelW, x1 = W - 20;
  const svg = sv("svg", {viewBox: `0 0 ${W} ${H}`, role: "img"});
  const grid = sv("g", {class: "grid"});
  for (let k = 0; k <= max; k += 1) { const x = x0 + (x1 - x0) * k / max; grid.append(sv("line", {x1: x, x2: x, y1: top, y2: H - 20, "stroke-opacity": k % 2 ? .45 : 1})); if (k % 2 === 0 || max <= 6) { const t = sv("text", {x, y: H - 6, "text-anchor": "middle"}); t.textContent = k; grid.append(t); } }
  svg.append(grid);
  rows.forEach((r, i) => {
    const y = top + i * rowH + 13;
    const lt = sv("text", {x: x0 - 8, y: y + 4, "text-anchor": "end"}); lt.textContent = r.label; svg.append(lt);
    const groups = {};
    r.points.forEach(p => { (groups[p.v] = groups[p.v] || {v: p.v, pts: []}).pts.push(p); });
    Object.values(groups).forEach(gp => {
      const x = x0 + (x1 - x0) * gp.v / max, ok = gp.pts.filter(p => !p.crash).length, bad = gp.pts.length - ok;
      gp.crash = bad > 0;
      if (!ok) { const t = sv("text", {x, y: y + 5, "text-anchor": "middle", "data-lab-style": "fill-var-critical-font-size-15px-font-weight-900", class: "mark"}); t.textContent = "✕"; svg.append(t); }
      else svg.append(sv("circle", {cx: x, cy: y, r: 5, fill: "var(--series-1)", stroke: "#0e1e33", "stroke-width": 2, class: "mark"}));
      if (ok && bad) { const c = sv("text", {x, y: y - 6, "text-anchor": "middle", "data-lab-style": "fill-var-critical-font-size-11px-font-weight-900"}); c.textContent = "✕"; svg.append(c); }
      if (ok > 1) { const t = sv("text", {x: x + 8, y: y + 4, class: "lbl", "font-size": 10}); t.textContent = "×" + gp.pts.length; svg.append(t); }
      const hit = sv("circle", {cx: x, cy: y, r: 11, class: "hit"}); svg.append(hit);
      if (tipFn) bindTip(hit, tipFn(r, gp));
    });
  });
  container.append(svg);
}
function charts() {
  const rows = EV.agents.map(a => { const rs = EV.runs.filter(r => r.agent === a.id); return {id: a.id, label: a.label, rs, value: rs.reduce((s, r) => s + r.metrics.overall, 0) / rs.length}; });
  hbar($("#c-score"), rows, {tipFn: r => () => `<b>${esc(r.label)}</b><div class="r">pass ${r.rs.filter(x => x.metrics.status === "pass").length}/8 · partial ${r.rs.filter(x => x.metrics.status === "partial").length} · crash ${r.rs.filter(x => x.metrics.status === "crash").length}</div>`});
  // breakdown heat table
  const keys = ["selection", "execution", "answer", "safety"];
  const tb = el("table", {class: "matrix", "data-lab-style": "min-width-0"});
  tb.append(el("thead", {}, el("tr", {}, el("th", {text: "Agent"}), ...keys.map(k => el("th", {scope: "col", text: k})))));
  const body = el("tbody");
  rows.forEach(r => { const tr = el("tr", {}, el("th", {scope: "row", text: r.label})); keys.forEach(k => { const v = r.rs.reduce((s, x) => s + x.metrics[k], 0) / r.rs.length; const td = el("td", {class: "cell", style: `background:${rampColor(v)};color:${v >= .7 ? "#07121f" : "#edf5ff"};cursor:default;height:30px`}, el("span", {class: "v", text: fmt(v)})); bindTip(td, `<b>${esc(r.label)}</b> · ${k}<div class="r">mean over 8 scenarios: ${fmt(v, 3)}</div>`); tr.append(td); }); body.append(tr); });
  tb.append(body); $("#c-break").append(tb);
  // depth
  const maxDepth = Math.max(4, ...EV.runs.map(r => r.nodes.length));
  dotPlot($("#c-depth"), rows.map(r => ({label: r.label, points: r.rs.map(x => ({v: x.nodes.length, sc: x.scenario, crash: !!x.crashed, path: x.nodes}))})), {max: maxDepth + (maxDepth % 2), tipFn: (r, g) => `<b>${esc(r.label)}</b> · ${g.v} node executions${g.crash ? " · graph raised" : ""}<div class="r">${g.pts.map(p => `${p.sc}: ${esc(p.path.join(" → "))}`).join("<br>")}</div>`});
  // calls
  const cr = rows.map(r => ({label: r.label, proposed: r.rs.reduce((s, x) => s + x.metrics.tool_calls, 0), executed: r.rs.reduce((s, x) => s + x.metrics.tool_messages, 0), errors: r.rs.reduce((s, x) => s + x.metrics.tool_errors, 0)}));
  groupedHBar($("#c-calls"), cr, [{key: "proposed", label: "proposed", color: "var(--series-1)"}, {key: "executed", label: "executed", color: "var(--series-2)"}], {max: Math.ceil((Math.max(...cr.map(r => Math.max(r.proposed, r.executed))) + 1) / 4) * 4, tipFn: r => `<b>${esc(r.label)}</b><div class="r">proposed ${r.proposed} · executed ${r.executed} · of which errors ${r.errors}</div>`});
  // MCP round trips
  const mrows = [];
  (DATA.mcp_sdk || []).forEach(t => t.calls.forEach(c => mrows.push({label: `${c.tool}(${Object.values(c.args).join(", ").trim() || "␣"})`, value: c.ms, color: t.transport === "stdio" ? "var(--series-1)" : "var(--series-2)", t, c})));
  const mc = $("#c-mcp");
  mc.append(el("div", {class: "legend", "data-lab-style": "margin-0-0-6px"}, el("span", {html: `<span class="sw" data-lab-style="background-var-series-1"></span>STDIO stock server`}), el("span", {html: `<span class="sw" data-lab-style="background-var-series-2"></span>Streamable HTTP news server`})));
  const mmax = Math.ceil(Math.max(...mrows.map(r => r.value)) / 20) * 20 || 20;
  hbar(mc, mrows, {max: mmax, labelW: 190, fmtV: v => `${fmt(v, 1)} ms`, tipFn: r => `<b>${esc(r.c.tool)}</b> via ${esc(r.t.transport)}<div class="r">args ${esc(JSON.stringify(r.c.args))}</div><div data-lab-style="margin-top-4px">${r.c.is_error ? '<span data-lab-style="color-ffb4b4">isError</span> · ' : ""}${esc(r.c.text)}</div><div class="r">${fmt(r.c.ms, 2)} ms</div>`});
  // latency
  const liveIds = new Set(EV.agents.filter(a => a.live).map(a => a.id));
  const lr = rows.filter(r => !liveIds.has(r.id)).map(r => ({label: r.label, value: median(r.rs.map(x => x.ms)), rs: r.rs}));
  const eng = DATA.engine || {};
  $("#lat-sub").textContent = `Wall time of one graph run${eng.runner === "GitHub Actions" ? " on a GitHub Actions runner" : ""}${eng.langgraph ? ` with langgraph ${eng.langgraph}` : " (LangGraph stand-in: treat as relative)"}. MCP agents include real MCP round trips.${liveIds.size ? " Live-model agents are left out: their time is dominated by model API calls." : ""}`;
  const lmax = Math.max(...lr.map(r => r.value)); const step = lmax > 4 ? 2 : 1;
  hbar($("#c-lat"), lr, {max: Math.ceil(lmax / step) * step || 1, fmtV: v => `${fmt(v, 2)} ms`, tipFn: r => `<b>${esc(r.label)}</b><div class="r">median ${fmt(r.value, 3)} ms · min ${fmt(Math.min(...r.rs.map(x => x.ms)), 3)} · max ${fmt(Math.max(...r.rs.map(x => x.ms)), 3)}</div>`});
  // tiles
  const all = EV.runs, pass = all.filter(r => r.metrics.status === "pass").length, crash = all.filter(r => r.crashed).length;
  const cli = DATA.cli || [], cliOk = cli.filter(c => c.expect_failure ? c.exit !== 0 : c.exit === 0).length;
  const tools = (DATA.mcp_sdk || []).reduce((s, t) => s + t.tools.length, 0);
  const tiles = [["Evaluated runs", all.length, `${EV.agents.length} agents × ${EV.scenarios.length} scenarios`], ["Passed", `${pass}`, `${Math.round(pass / all.length * 100)}% of runs scored 1.00`], ["Graph crashes", crash, crash ? [...new Set(all.filter(r => r.crashed).map(r => agentOf(r.agent).label))].join(", ") : "none"], ["MCP tools", tools, "3 over STDIO · 2 over HTTP"], ["CLI commands", `${cliOk}/${cli.length}`, "behaved as documented"]];
  tiles.forEach(([k, n, d]) => $("#tiles").append(el("div", {class: "tile"}, el("div", {class: "k", text: k}), el("div", {class: "n", text: n}), el("div", {class: "d", text: d}))));
}

/* ---------------- MCP + CLI ---------------- */
function mcpPanels() {
  (DATA.mcp_sdk || []).forEach(t => {
    const p = $(t.transport === "stdio" ? "#mcp-stdio" : "#mcp-http");
    p.append(el("h2", {text: t.transport === "stdio" ? "STDIO MCP server · stock_market_mcp_server.py" : "Streamable HTTP MCP server · news_mcp_server.py"}));
    p.append(el("p", {class: "sub", text: t.transport === "stdio" ? "The client spawned the server as a child process (as the README says), then listed and called its tools." : "Started separately on 127.0.0.1:8000/mcp, then the client connected over HTTP."}));
    const dl = el("dl", {class: "kv"});
    [["server", t.server], ["protocol", t.protocol], ["connect + initialize", `${fmt(t.connect_ms, 1)} ms`], ["tools/list", `${fmt(t.list_ms, 2)} ms`]].forEach(([k, v]) => dl.append(el("dt", {text: k}), el("dd", {class: "mono", text: v})));
    p.append(dl);
    const tt = el("table", {class: "tbl", "data-lab-style": "margin-top-10px"});
    tt.append(el("thead", {}, el("tr", {}, el("th", {text: "Tool"}), el("th", {text: "Description"}))));
    const tb = el("tbody"); t.tools.forEach(x => tb.append(el("tr", {}, el("td", {class: "mono", text: `${x.name}(${x.params.join(", ")})`}), el("td", {text: x.description})))); tt.append(tb); p.append(tt);
    const ct = el("table", {class: "tbl", "data-lab-style": "margin-top-10px"});
    ct.append(el("thead", {}, el("tr", {}, el("th", {text: "tools/call"}), el("th", {text: "Result"}), el("th", {text: "ms"}))));
    const cb = el("tbody"); t.calls.forEach(c => cb.append(el("tr", {}, el("td", {class: "mono", text: `${c.tool}(${JSON.stringify(c.args)})`}), el("td", {class: c.is_error ? "err" : "", text: (c.is_error ? "isError · " : "") + c.text}), el("td", {class: "num", text: fmt(c.ms, 1)})))); ct.append(cb); p.append(ct);
  });
  const list = $("#term-list"), out = $("#term-out"), note = $("#term-note");
  (DATA.cli || []).forEach((c, i) => {
    const ok = c.expect_failure ? c.exit !== 0 : c.exit === 0;
    const b = el("button", {class: "term-item", type: "button", "aria-pressed": "false"}, el("span", {class: "ec " + (ok ? "ok" : "bad"), text: ok ? "✓" : "✕"}), el("span", {text: c.cmd.replace("python MCP_Langchain_Langraph_Tools.py", "…Tools.py")}));
    b.addEventListener("click", () => {
      list.querySelectorAll(".term-item").forEach(x => x.setAttribute("aria-pressed", x === b));
      out.textContent = `$ ${c.cmd}\n${c.output}\n\n[exit ${c.exit} · ${fmt(c.ms, 0)} ms]`;
      const n = CLI_NOTES.find(([re]) => re.test(c.cmd));
      const obs = c.expect_failure ? (c.exit !== 0 ? `exit ${c.exit}: it fails, as documented.` : `exit 0: it does not fail. The MCP adapter logs the refused connection, skips the news server and carries on with the STDIO tools.`) : `exit ${c.exit} in ${fmt(c.ms, 0)} ms.`;
      note.innerHTML = n ? `<div class="src">${esc(n[1])}</div><p data-lab-style="margin-4px-0">${esc(n[2])}</p><p data-lab-style="margin-4px-0"><strong>Observed:</strong> ${esc(obs)}</p>` : "";
    });
    list.append(b); if (i === 0) b.click();
  });
}

/* ---------------- notes ---------------- */
function notes() {
  FLOW.forEach(([d, o]) => $("#flow").append(el("li", {html: `${esc(d)}<br><span data-lab-style="color-var-dim">↳ ${esc(o)}</span>`})));
  const cli = DATA.cli || [], ran = re => cli.find(c => re.test(c.cmd) && c.exit === 0);
  const tests = [
    ["ok", "Install requirements.txt in a fresh Python 3.12 environment", (DATA.engine || {}).langgraph ? `Ran on Python ${(DATA.engine || {}).python}${(DATA.engine || {}).runner === "GitHub Actions" ? " (GitHub Actions)" : ""} with langgraph ${(DATA.engine || {}).langgraph}, langchain ${(DATA.engine || {}).langchain || "?"} and MCP SDK ${(DATA.engine || {}).mcp}.` : `Ran on Python ${(DATA.engine || {}).python}. MCP SDK ${(DATA.engine || {}).mcp} was installed; langgraph/langchain were not installable there, so their API was supplied by a minimal stand-in.`, !((DATA.engine || {}).langgraph) ? "warn" : "ok"],
    ["ok", "Run --topic tools and --topic graph-prebuilt", ran(/--topic tools$/) && ran(/graph-prebuilt/) ? "Both exit 0; outputs under MCP & CLI runs." : "Not completed."],
    ["ok", "Run --topic mcp-client and confirm STDIO tool discovery", (() => { const c = ran(/mcp-client$/); if (!c) return "Not completed."; const tl = (c.output.match(/STDIO stock server tools: \[([^\]]*)\]/) || [])[1], sr = (c.output.match(/STDIO sample tool result: (\S+)/) || [])[1]; return `${tl ? tl.replace(/'/g, "") + " discovered" : "Tools discovered"}${sr ? `; sample call returned ${sr}` : ""}.`; })()],
    ["ok", "Launch news_mcp_server.py and rerun with --include-news", ran(/mcp-client --include-news$/) ? `Five tools across both transports. ${((DATA.cli || []).find(c => c.expect_failure) || {}).exit ? "Without the server the same command fails, as documented." : "Without the server the command still exits 0: the adapter skips the unreachable server."}` : "Not completed."],
    (() => { const lv = (DATA.engine || {}).live || {}; return /ran|carried/.test(lv.status || "") ? ["ok", "Verify --live calls only after granting API access", `Live tool calling verified on ${lv.model}${lv.at ? " (" + lv.at + ")" : ""}, using a temporary API key that was then removed. Tavily was not tested.`] : ["skip", "Verify --live calls only after granting API access", "Skipped: no API key was provided, and live calls can incur charges."]; })()
  ];
  tests.forEach(([st, t, d, override]) => { const s = override || st; $("#itests").append(el("div", {class: "check"}, el("span", {class: "ic " + s, text: s === "ok" ? "✓" : s === "warn" ? "!" : "–", "aria-label": s}), el("div", {}, el("strong", {text: t}), el("p", {text: d})))); });
  const find = (a, s) => runOf(a, s) || {metrics: {}};
  const F = [
    ["Only the looping graphs produce a final answer.", `graph-prebuilt and mcp-graph end on a ToolMessage (answer check 0 on every tool scenario); react and supervisor return to the model and finish with an AIMessage.`],
    ["Invalid quantities produce an error result.", "For S6, local graph tools and MCP tools both return the reason the quantity was rejected. ReAct uses the error to explain the result."],
    ["Each specialist summarizes its own latest tool calls.", "The news specialist omits earlier stock-tool output, and distinct tool-call IDs link each result to its request."],
    ["Multi-agent routing costs depth, not correctness.", `On S8 the supervisor takes ${find("supervisor", "S8").nodes ? find("supervisor", "S8").nodes.length : 9} node executions against ${find("mcp-react", "S8").nodes ? find("mcp-react", "S8").nodes.length : 3} for single-agent ReAct with the same five tools. Both score 1.00; the supervisor keeps each agent's tool access scoped.`],
    ["Guardrails hold on every path.", "No run produced a “would buy 0” trade or advice text. Invalid quantities are rejected by simulated_trade on both the local and MCP paths."]
  ];
  F.forEach(([t, d]) => $("#findings").append(el("div", {class: "check"}, el("span", {class: "ic ok", text: "i", "aria-hidden": "true", "data-lab-style": "background-var-cyan"}), el("div", {}, el("strong", {text: t}), el("p", {html: d})))));
  const D = [
    ["CLI and local setup", "Use the Python commands in the README. Install the dependencies into the interpreter that launches both MCP processes."],
    ["HTTP server setup", "Start the local news server before using --include-news. The CLI reports when the requested tools are unavailable."]
  ];
  D.forEach(([t, d]) => $("#docnotes").append(el("div", {class: "check"}, el("span", {class: "ic ok", text: "i", "aria-hidden": "true"}), el("div", {}, el("strong", {text: t}), el("p", {text: d})))));
}

/* ---------------- height reporting for the parent iframe ---------------- */
function reportHeight() { try { if (window.parent !== window) window.parent.postMessage({type: "agent-lab:height", height: Math.ceil(document.querySelector(".lab").getBoundingClientRect().height) + 2}, "*"); } catch (e) { /* not embedded */ } }

if (!EV.runs.length) { document.querySelector(".lab").append(el("p", {class: "panel", text: "No lab data found. Run: python lab/run_agent_lab.py"})); return; }
badges();
const show = tabs();
runner(); matrix(show); charts(); mcpPanels(); notes();
if ("ResizeObserver" in window) new ResizeObserver(reportHeight).observe(document.body); reportHeight();
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => setTimeout(reportHeight, 30)));
const q = new URLSearchParams(location.search); if (q.get("agent") && agentOf(q.get("agent"))) selectAgent(q.get("agent")); if (q.get("autoplay") === "1") play();
})();
