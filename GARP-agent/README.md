# Quantamental Research Agent 📈
**A Multi-Agent Framework for GARP Discovery & Forensic Analysis**

## 🎯 Overview
This project implements an Agentic AI workflow to identify and analyze companies with high growth potential and reasonable valuations (**GARP**). It enforces a strict separation between quantitative calculation (handled by **Python/NumPy**) and qualitative reasoning (handled by **LLMs** via OpenRouter).

## 🛠️ The Seven-Step Methodology

### 1. Quantitative Screener (The Gatekeeper) 🔍
* **Module**: `modules/screener.py`
* **Geography Selection**: System prompts for **US** or **India** to define the `yfinance` search universe.
* **Filter Logic**: 
    * Revenue Growth $\ge 19\%$ YoY (last two quarters).
    * EPS Growth $\ge 19\%$ YoY (last two quarters).
    * PE Expansion $\le 10\%$ (last three quarters).
* **Data Export**: Candidates passing the filter are exported to a `.csv` spreadsheet.
* **Human-in-the-Loop**: System pauses for the user to input a specific **Ticker** to begin deep-dive research.

### 2. Visualization & Technicals 📊
* **Module**: `modules/charts.py`
* **Chart A**: Dual-axis plot of TTM EPS vs TTM PE, including a mean PE line and $\pm 1, 2$ Standard Deviation bands.
* **Chart B**: Share price with RSI ($40-60$ range) plotted below.
* **Output**: Interactive Plotly figures embedded into the final report.

### 3. Sentiment & News Flow 📰
* **Module**: `modules/analyst.py`
* **Task**: Agentic search for recent news and drivers of growth.
* **Requirement**: Succinct summary with cited sources, specifically flagging major "red flags."

### 4. Competitive Moat Analysis 🏰
* **Module**: `modules/analyst.py`
* **Classification**: Categorized as **Moat**, **No Moat**, or **Limited Moat**.
* **Logic**: Analysis of pricing power, network effects, and barriers to entry.

### 5. Management & Affiliations 🤝
* **Module**: `modules/analyst.py`
* **Focus**: Cross-holdings and track records for ethics, growth, and profitability.

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
* **Final Output**: Generates a standalone **HTML Investment Memo** containing the analysis, score, and interactive charts.

## 🏗️ Technical Architecture
* **Execution Engine**: Python 3.10+
* **Model**: `gpt-oss-120b` (via OpenRouter)
* **Math & Data**: `numpy`, `pandas`, `scipy`, `yfinance`
* **Visualization**: `plotly`
* **Reporting**: `jinja2` (HTML templating)
* **Security**: API keys managed via `.env`.

## 📂 Project Structure
/GARP-agent
├── main.py             # The Conductor (runs the workflow)
├── .env                # Secret keys (ignored by Git)
├── requirements.txt    # Library list
└── modules/
    ├── screener.py     # Step 1: Growth Filter
    ├── charts.py       # Step 2: Visuals (Plotly)
    ├── analyst.py      # Steps 3, 4, 5: Qualitative Brain
    ├── forensic.py     # Step 6: Earnings Quality
    └── reporter.py     # Step 7: Scoring & HTML Publisher

## ⚖️ License
**© 2026 All Rights Reserved.**
This software and its underlying logic are proprietary. No part of this framework may be reproduced or distributed without explicit permission.