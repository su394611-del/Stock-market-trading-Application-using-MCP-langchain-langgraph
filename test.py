import unittest

from MCP_Langchain_Langraph_Tools import fake_price, interpret_demo_request, simulated_trade


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
