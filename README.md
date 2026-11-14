# Stock-market-trading-Application-using-MCP-langchain-langgraph
Stock market trading application using MCP-LangChain-LangGraph
#  LangChain Tools, MCP and LangGraph/ReAct
 LangChain Tools, MCP and LangGraph/ReAct production-grade example or demonstration.
 **`Model Context Protocol (MCP).ipynb`**. Includes a standalone Python program, a guided Jupyter Notebook, a STDIO MCP server, a Streamable HTTP MCP server, a dependency list, and a Windows launcher.

> **Safety and accuracy:** This course uses simulated stock quotes, synthetic market news, and simulated buy/sell operations. It **never** places brokerage orders and does not offer investment advice. The *offline* model is a small, deterministic, rule-based demonstration, **not ** a real decoder-only  LLM. Using `--live` enables OpenAI API requests (costs may apply).

## Files

| File | Purpose |
|---|---|
| `MCP_Langchain_Langraph_Tools.py` | Standalone CLI  |
| `MCP_Langchain_Langraph_Tools.ipynb` | Step-by-step notebook for VS Code/Jupyter/Colab |
| `stock_market_mcp_server.py` | MCP stock server with **STDIO** transport |
| `news_mcp_server.py` | MCP news server with **Streamable HTTP** transport |
| `requirements.txt` | Python package dependencies |
| `.env.example` | Optional API-key template; no real credentials are included |
| `run_windows.cmd` | Windows double-click launcher (requires installed packages) |
## Requirements

- **Python 3.12** recommended (Windows 11, Linux, macOS, Google Colab compatible subject to package support).
- A network connection is required for initial package installation, for downloading packages and when using paid APIs. No GPU/CUDA required.
- The **Python MCP SDK 2.x** is used by default. The included server files use the SDK 2.x import path.
- The **LangChain 1.4+ built-in MCPAdapter** is used. The original notebook's `MultiServerMCPClient` came from a legacy package whose current 0.3.2 release requires MCP SDK `<2`, so **do not combine `langchain-mcp-adapters` with MCP SDK 2**.
- OpenAI and Tavily API keys are only required for the explicitly enabled live examples.

### Installation on Windows PowerShell (recommended)

Extract the ZIP to a folder, then:

```powershell
cd C:\HF_Projects\1\2
conda create -n mcp_langchain_langgraph-new python=3.12 -y
conda activate mcp_langchain_langgraph-new
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or use a built-in virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `Activate.ps1` is blocked by PowerShell policy, run `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and call that interpreter directly instead of changing system security settings.

### Jupyter / Google Colab

Upload/keep all project files in one directory. Install dependencies in a notebook cell:

```python
%pip install -r requirements.txt
```

Restart the kernel if prompted. Open `MCP_Langchain_Langraph_Tools.ipynb` and run in order. Use `await run_topic(...)` inside Jupyter cells—**not** `asyncio.run(...)`—because the notebook already has an event loop. The files in this ZIP need to be available next to the notebook.

For Colab, upload/extract the ZIP into the Colab runtime and set the working directory before running the cells. Remote API examples require internet access and your own credentials.

## Run commands

```powershell
python MCP_Langchain_Langraph_Tools.py --list
python MCP_Langchain_Langraph_Tools.py --self-test
python test.py
python MCP_Langchain_Langraph_Tools.py --topic all
python MCP_Langchain_Langraph_Tools.py --topic tools
python MCP_Langchain_Langraph_Tools.py --topic tool-calling
python MCP_Langchain_Langraph_Tools.py--topic manual-execution
python MCP_Langchain_Langraph_Tools.py --topic graph-basic
python MCP_Langchain_Langraph_Tools.py --topic graph-manual-tools
python MCP_Langchain_Langraph_Tools.py --topic graph-conditional
python MCP_Langchain_Langraph_Tools.py--topic graph-prebuilt
python MCP_Langchain_Langraph_Tools.py --topic mcp-client
python MCP_Langchain_Langraph_Tools.py --topic mcp-graph
python MCP_Langchain_Langraph_Tools.py --topic react --graph-mermaid
```

Default `--topic all` runs **offline** demos for all topics. It uses a deterministic rule-based model rather than calling OpenAI. For LangChain and MCP topics, you must install the third-party packages. `--self-test` and `test_chapter11.py` do not require any third-party packages.

### Start the MCP servers

**STDIO:** The client automatically launches `stock_market_mcp_server.py` as a child process. **Do not** manually start this one in a terminal to use the client.

**Streamable HTTP:** Open another terminal, activate the same environment, and start:

```powershell
python news_mcp_server.py
```

The HTTP server listens on **`127.0.0.1:8000/mcp`**. Keep that terminal running. From a second terminal:

```powershell
MCP_Langchain_Langraph_Tools.py --topic mcp-client --include-news
MCP_Langchain_Langraph_Tools.py --topic mcp-graph --include-news
MCP_Langchain_Langraph_Tools.py--topic react --include-news
```

`--include-news` will fail if the HTTP server is not listening. The HTTP example intentionally binds to localhost, not an external interface, and has no authentication; **do not expose it publicly**.

### Live OpenAI tool calling (optional)

Copy `.env.example` to `.env` in the same folder:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set your `OPENAI_API_KEY` without printing it or committing it to Git. If you intend to try Tavily, also set `TAVILY_API_KEY`.

```powershell
MCP_Langchain_Langraph_Tools.py --topic tool-calling --live
MCP_Langchain_Langraph_Tools.py --topic graph-prebuilt --live
MCP_Langchain_Langraph_Tools.py --topic tavily --live
MCP_Langchain_Langraph_Tools.py --topic react --live
```

The default model is `gpt-5-mini` to match the supplied source. Change it with `--model YOUR_MODEL_ID` if your account supports a different tool-capable model. Live LLM calls may incur charges. `--topic tavily` demonstrates binding the search tool; its code does not automatically execute a tool call proposed by the LLM.

## Topics mapped from the source notebook

| Source section | Converted example |
|---|---|
| Tool creation and direct invocation | `tools` |
| LLM tool calls, several tool calls, no-tool case | `tool-calling` |
| Manual tool execution / `ToolMessage` / tool dictionaries | `manual-execution` |
| LangGraph LLM node | `graph-basic` |
| LangGraph manual tool-execution node | `graph-manual-tools` |
| Conditional routing | `graph-conditional` |
| `ToolNode` and `tools_condition` | `graph-prebuilt` |
| Tavily built-in search tool | `tavily` |
| STDIO + Streamable HTTP MCP servers, modern `MCPAdapter` (replacing `MultiServerMCPClient`) | `mcp-client` + supplied server files |
| LangGraph connecting to MCP tools | `mcp-graph` |
| ReAct agent loop | `react` |

## Corrections relative to the source notebook

1. **No broken environment assignments:** removed `os.environ["OPENAI_API_KEY"] = os.getenv(...)`, which raises when the key is missing; reading `.env` is optional and explicit.
2. **Correct return types:** `stock_price()` returns `float` rather than the incorrect `int` annotation from the first notebook cell.
3. **Self-contained MCP servers:** both server `.py` files that were only referenced in the notebook are now included.
4. **Standalone Python entry point:** CLI argument handling and `asyncio.run()` for MCP client calls; notebook uses `await` correctly.
5. **Safe tool demonstrations:** fixed example values; simulated trades; no brokerage connections.
6. **No execution-order dependency:** every section constructs its own tools and graph, rather than relying on earlier cells' global variables.
7. **No hidden external graph rendering:** `--graph-mermaid` prints Mermaid source locally rather than downloading remote PNG graphics.
8. **Tool execution and routing:** `ToolNode`, `tools_condition`, `MessagesState`, and `ToolMessage` patterns use current LangGraph and LangChain approaches.
11. **Critical dependency compatibility:** replaced legacy `langchain-mcp-adapters` with the native `langchain.mcp.MCPAdapter` introduced in LangChain 1.4, because the old adapters require `mcp<2` but the current `MCPServer` requires SDK v2. `get_tools(server_name=...)` is replaced with `list_tools()` and filtering server-prefixed tool names.
9. **Explicit HTTP server setup:** optional Streamable HTTP at `http://127.0.0.1:8000/mcp`; example is not exposed to the public internet.
10. **Offline first:** a deterministic tool-calling model lets you explore graph control flow without paying for APIs; it is *not* real AI inference.

## Known limitations

- This is a ** production-grade runnable Python program**, not a native compiled Windows `.exe`. The included `.cmd` launcher can be double-clicked on Windows after installing dependencies. Native EXE packaging with PyInstaller is outside the scope of this conversion and can require bundling server scripts and the Python packages.
- The `--self-test` only tests standard-library functionality; it **does not verify** live OpenAI, Tavily, HTTP MCP interoperability, or third-party LangGraph API behavior.
- This package was syntax-checked and passed standard-library tests, but complete third-party integration execution was not available in the conversion environment due to missing packages.
- The demonstration parser is deliberately limited; example prompts are selected to be deterministic. Use `--live` for real LLM tool-call selection.
- The MCP client spawns a fresh STDIO session for tool discovery/invocation. The HTTP news server must be started separately.
- Demo news and prices are *not* real. For live news, implement a trusted source with appropriate authentication and attribution.

## References used for compatibility checks

- MCP SDK v2 running servers: https://py.sdk.modelcontextprotocol.io/run/
- MCP SDK migration from FastMCP to MCPServer: https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/migration.md
- Modern native MCP adapter: https://www.langchain.com/blog/mcp-in-langchain-stateless-protocol-elicitation-and-more
- Old adapter compatibility warning: https://github.com/langchain-ai/langchain-mcp-adapters/releases
- LangGraph `ToolNode`: https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolNode
- LangGraph `tools_condition`: https://reference.langchain.com/python/langgraph.prebuilt/tool_node/tools_condition
This project is licensed under the Apache License, Version 2.0.© 2026 Sudhanshu Shukla · Licensed under the Apache License, Version 2.0
