from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# ✅ Initialize Gemini models
deep_think_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

quick_think_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ✅ Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = deep_think_model         # Slower, more capable model
config["quick_think_llm"] = quick_think_model       # Faster, cheaper model
config["max_debate_rounds"] = 1

# ✅ Set provider to Google Gemini
config["llm_provider"] = "google"

# Configure data vendors (default uses yfinance and alpha_vantage)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "alpha_vantage",
    "news_data": "alpha_vantage",
}

# ✅ Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# ✅ Forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Optional memory reflection
# ta.reflect_and_remember(1000)
