import unittest
from importlib.util import find_spec

from MCP_Langchain_Langraph_Tools import (
    DemoToolCallingModel, build_graph, fake_price, interpret_demo_request, make_tools, simulated_trade,
)


class MCP_LangChain_LangGraph(unittest.TestCase):
    def test_known_price_is_fixed_demo_data(self):
        self.assertEqual(fake_price("apple"), 175.0)
        self.assertEqual(fake_price("Microsoft"), 320.0)

    def test_unknown_price_is_placeholder_not_live(self):
        self.assertEqual(fake_price("Unknown Corporation"), 100.0)

    def test_simulated_trade_cannot_place_an_order(self):
        message = simulated_trade("buy", "Apple", 3)
        self.assertIn("SIMULATION ONLY", message)
        self.assertIn("No trade was placed", message)

    def test_invalid_trades_rejected(self):
        for q in (0, -2, True, 0.5):
            with self.subTest(q=q), self.assertRaises(ValueError):
                simulated_trade("sell", "Apple", q)

    def test_multi_tool_request(self):
        result = interpret_demo_request("Sell 25 Apple shares and purchase 10 Microsoft stocks",
                                        ["sell_stock", "buy_stock"])
        self.assertEqual([r["name"] for r in result], ["sell_stock", "buy_stock"])
        self.assertEqual([r["args"]["qty"] for r in result], [25, 10])

    def test_pe_ratio_no_tool(self):
        self.assertFalse(interpret_demo_request("Explain the P/E ratio", ["stock_price"]))

    def test_stock_price_tool(self):
        result = interpret_demo_request("What is the price of IBM stocks?", ["stock_price"])
        self.assertEqual(result[0]["name"], "stock_price")

    def test_prefixed_mcp_tools(self):
        # Native LangChain MCPAdapter prefixes tools with server names.
        result = interpret_demo_request("What is the price of Apple stock?", ["stock_stock_price"])
        self.assertEqual(result[0]["name"], "stock_stock_price")
        result = interpret_demo_request("Sell 2 Apple shares", ["stock_sell_stock"])
        self.assertEqual(result[0]["name"], "stock_sell_stock")

    def test_sentiment_tool(self):
        result = interpret_demo_request("What are market sentiments of Tesla?", ["market_sentiment"])
        self.assertEqual(result[0]["name"], "market_sentiment")


@unittest.skipUnless(find_spec("langgraph") and find_spec("langchain_core"), "requires LangGraph")
class GraphToolErrors(unittest.TestCase):
    def test_invalid_trade_returns_error_with_reason(self):
        from langchain_core.messages import HumanMessage, ToolMessage

        for variant in ("manual-tools", "conditional", "prebuilt", "react"):
            with self.subTest(variant=variant):
                graph = build_graph(DemoToolCallingModel(), make_tools(), variant=variant)
                messages = graph.invoke({"messages": [HumanMessage(content="Buy 0 shares of IBM")]})["messages"]
                tool_message = next(msg for msg in messages if isinstance(msg, ToolMessage))
                self.assertEqual(tool_message.status, "error")
                self.assertIn("positive integer", tool_message.content)
                if variant == "react":
                    self.assertIn("reported an error", messages[-1].content)

    def test_tool_schema_rejects_boolean_quantity(self):
        from pydantic import ValidationError

        buy_stock = next(tool for tool in make_tools() if tool.name == "buy_stock")
        with self.assertRaises(ValidationError):
            buy_stock.invoke({"name": "IBM", "qty": True})

    def test_demo_response_uses_only_latest_tool_batch(self):
        from langchain_core.messages import AIMessage, ToolMessage

        response = DemoToolCallingModel().invoke([
            ToolMessage(content="135.0", tool_call_id="stock"),
            AIMessage(content="Stock lookup complete"),
            ToolMessage(content="news is unavailable", tool_call_id="news", status="error"),
        ])
        self.assertEqual(response.content, "Demo tool execution reported an error: news is unavailable")

    def test_demo_uses_a_new_call_id_after_an_earlier_batch(self):
        from langchain_core.messages import AIMessage, HumanMessage

        model = DemoToolCallingModel(make_tools())
        prior_call = model.invoke("What is the price of IBM?")
        new_call = model.invoke([prior_call, AIMessage(content="Price checked"),
                                 HumanMessage(content="What is the price of Apple?")])
        self.assertNotEqual(prior_call.tool_calls[0]["id"], new_call.tool_calls[0]["id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
