# Wolf on Wall Street - AI Stock Analyst & Risk Management

This project is a Multi-Agent System (MAS) built to analyze the stock market. Instead of using a single AI model that might hallucinate or be biased, we created a team of AI agents that act like a real trading firm. They gather data, debate with each other (Bull vs. Bear), and assess risks before making a final decision.

It uses Google's Agent Development Kit (ADK), LangGraph, and the Gemini 2.0 Flash model.

## Key Features

* **Adversarial Debate:** We implemented a "Debate" architecture where a Bull Agent and a Bear Agent argue about a trade. This helps reduce bias and catches potential risks that a single agent might miss.
* **Memory & Learning (RAG):** The system uses ChromaDB to store vector embeddings of past trades. Before making a new decision, it looks at similar past situations to learn from previous mistakes.
* **Risk Management Triad:** A voting system involving three distinct personalities—Aggressive, Conservative, and Neutral. They review the trade plan and vote on the final Buy/Sell/Hold signal.
* **Cyclic Workflow:** Unlike linear chatbots, this uses a graph-based workflow (LangGraph). Agents can loop back to get more information or continue debating if they haven't reached a consensus.
* **Multi-Source Data:** It pulls data from Alpha Vantage (technical data), financial statements, and scrapes Google News for sentiment analysis.

## How It Works

The system follows a 4-step process:

1.  **Intelligence Phase:** The Analyst Team gathers market data, news, and technical indicators.
2.  **Strategic Debate:** The Bull and Bear agents debate the findings.
3.  **Execution Phase:** If a consensus is reached, a Research Manager creates a trade plan.
4.  **Risk Phase:** The Risk Triad critiques the plan and the Risk Manager makes the final decision.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/PranavKanumuri/Wolf-on-Wall-Street.git](https://github.com/PranavKanumuri/Wolf-on-Wall-Street.git)
    cd Wolf-on-Wall-Street
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up your API keys:
    Create a `.env` file in the main folder and add your keys:
    ```env
    GOOGLE_API_KEY=your_gemini_key
    ALPHAVANTAGE_API_KEY=your_alpha_vantage_key
    ```

## Usage

Run the main script:
```bash
python main.py
```

Run the cli script:
```bash
cd cli
python main.py
```
