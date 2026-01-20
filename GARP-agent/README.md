# Quantamental Research Agent 📈
**A Multi-Agent Framework for GARP Discovery & Forensic Analysis**

## 🏗️ Technical Architecture
Framework: Agno (formerly Phidata).
LLM: Ollama (llama3.2:3b).
Embeddings: nomic-embed-text.
Database: LanceDB (Local vector storage).
Search: DuckDuckGo (via ddgs).

## 🚀 Installation & Setup
### 1. Prerequisites
Python 3.10+

Ollama: Installed and running - 
ollama run llama3.2:3b
ollama run nomic-embed-text

### 2. Installation
pip install -r requirements.txt

### 3. Configuration
Ensure your directory structure includes the static inputs - list of yfinance tickers for US and Indian stocks.

### 4. Usage
Run the main pipeline:
Follow the CLI prompts to select a region (US/India) and pick a stock.

## 🎯 Overview
This project implements an Agentic AI workflow to identify and analyze companies with high growth potential and reasonable valuations (**GARP**). It enforces a strict separation between quantitative calculation (handled by **Python/NumPy**) and qualitative reasoning (handled by **LLMs** via OpenRouter).

## 🛠️ The Seven-Step Methodology

### 1. Quantitative Screener (The Gatekeeper) 🔍
* **Module**: `modules/screener.py`
* **Geography Selection**: System prompts for **US** or **India** to define the `yfinance` search universe.
* **Filter Logic** (tweak as desired): 
    * Revenue Growth $\ge ???\%$ YoY (latest quarter and annual period).
    * EPS Growth $\ge ???\%$ YoY (latest quarter and annual period).
    * PE Expansion $\le ???\%$ QoQ (latest quarter on TTM basis).
* **Data Export**: Candidates passing the filter are exported to a `.csv` spreadsheet.
* **Human-in-the-Loop**: System pauses for the user to input a specific **Ticker** to begin deep-dive research.

### 2. Visualization & Technicals 📊
* **Module**: `modules/charts.py`
* **Chart A**: Share price and PEG ratio with RSI ($40-60$ range) plotted below.
* **Chart B**: Dual-axis plot of TTM EPS vs TTM PE, including a mean PE line and $\pm 1, 2$ Standard Deviation bands.
* **Fundamentals**: Tabular metrics and ratios used in the analysis.
* **Output**: Interactive Plotly figures and tables embedded into the final report.

### 3. Sentiment & News Flow 📰
* **Module**: `modules/analyst.py`
* **Task**: Agentic search for recent news and drivers of growth.
* **Requirement**: Succinct summary with cited sources, specifically flagging major "red flags."
* **Agent Grounding**: Latest available annual report used to verify sell-side research hype with management commentary.

### 4. Competitive Moat Analysis 🏰
* **Module**: `modules/analyst.py`
* **Classification**: Categorized as **Moat**, **No Moat**, or **Limited Moat**.
* **Logic**: Analysis of pricing power, network effects, and barriers to entry.
* **Agent Grounding**: Latest available annual report used to verify sell-side research hype with management commentary.

### 5. Management & Affiliations 🤝
* **Module**: `modules/analyst.py`
* **Focus**: Cross-holdings and track records for ethics, growth, and profitability.
* **Agent Grounding**: Latest available annual report used to verify sell-side research hype with management commentary.

### 6. Forensic Earnings Quality 🧪
* **Module**: `modules/forensic.py`
* **RONW Thresholds**: $\ge 14\%$ (Emerging/India) or $\ge 7\%$ (Developed/US).
* **Solvency Check**: Debt Coverage $\ge 1.5$ or **Zero Debt**.
* **Agentic Normalization**: Agent scans notes to accounts for "exceptional items" to suggest a normalized EBIT.

### 7. Composite Scoring & Reporting 🏆
* **Module**: `modules/reporter.py`
* **Scoring Weights**:
    1.  News Flow Sentiment: $30\%$
    2.  Competitive Position: $20\%$
    3.  Management & Affiliation: $20\%$
    4.  Quality of Earnings: $30\%$
* **Final Output**: Generates a standalone **HTML Investment Memo** containing the analysis, score, and interactive charts, with an overall sentiment.

## 📂 Project Structure
```text
/GARP-agent
├── main.py             # The Conductor (runs the workflow)
├── .env                # Secret keys (ignored by Git)
├── .gitignore          # Contents to be ignored by Git
├── README.md           # This file
├── requirements.txt    # Library list
└── modules/
    ├── screener.py     # Step 1: Growth Filter
    ├── charts.py       # Step 2: Visuals (Plotly)
    ├── analyst.py      # Steps 3, 4, 5: Qualitative Brain
    ├── forensic.py     # Step 6: Earnings Quality
    └── reporter.py     # Step 7: Scoring & HTML Publisher
```

## ⚖️ License
**© 2026 All Rights Reserved.**
This software and its underlying logic are proprietary. No part of this framework may be reproduced or distributed without explicit permission.