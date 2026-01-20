import os
import glob
import time
import json
import re
import pandas as pd
import markdown 
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from tqdm import tqdm

# Agno (Phidata) Imports
from agno.agent import Agent
from agno.models.ollama import Ollama 

MODEL_ID = "llama3.2:3b"

# --- SCORING CONFIGURATION ---
SCORING_WEIGHTS = {
    "News": 3,
    "Moat": 2,
    "Management": 2,
    "Earnings Quality": 3
}

# --- STATE MANAGEMENT ---

class MemoState:
    def __init__(self, ticker):
        self.ticker = ticker
        self.market_data = ""
        self.forensic_data = ""
        self.valuation_facts = {}
        self.valuation_table = ""
        self.full_annexure_table = "" # Annexure 1 (JSON Data)
        self.scores = {}       # Raw integers from LLM
        self.scoring_data = {} # Computed totals & recommendation (Python)
        self.thesis = ""
        self.final_markdown = ""

# --- HELPER FUNCTIONS ---

def get_latest_file(ticker: str, prefix: str) -> Optional[str]:
    """Finds the most recent file in 'outputs/'."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    search_pattern = os.path.join(outputs_dir, f"{prefix}*{ticker}*")
    files = glob.glob(search_pattern)
    return max(files, key=os.path.getmtime) if files else None

def get_file_content(filepath: str) -> str:
    """Safely reads text content."""
    if not filepath or not os.path.exists(filepath): return "Data not available."
    with open(filepath, "r", encoding="utf-8") as f: return f.read()

def get_fundamental_json(ticker: str) -> Optional[str]:
    """Finds the fundamentals_{ticker}.json file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    filepath = os.path.join(outputs_dir, f"fundamentals_{ticker}.json")
    return filepath if os.path.exists(filepath) else None

def build_valuation_facts(ticker: str) -> Dict[str, Any]:
    """Reads JSON and builds context dictionary + markdown table for the Agent."""
    json_path = get_fundamental_json(ticker)
    if not json_path: return {"available": False}

    try:
        with open(json_path, 'r') as f: data = json.load(f)
        if not data: return {"available": False}

        df = pd.DataFrame(data)
        latest = df.iloc[-1]
        
        # Build Table String (Last 4 periods for the prompt context)
        history_df = df.tail(4).copy()
        cols = ["Report_Date_Official", "Close", "PE_Ratio", "PEG_Ratio", "Revenue_TTM", "Net_Margin_Pct", "RONW_Pct"]
        cols = [c for c in cols if c in history_df.columns]
        
        # Format for readability
        table_str = history_df[cols].to_markdown(index=False)

        return {
            "available": True,
            "table_str": table_str,
            "current_pe": latest.get("PE_Ratio"),
            "current_peg": latest.get("PEG_Ratio"),
            "price": latest.get("Close")
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

def build_full_annexure(ticker: str) -> str:
    """Reads JSON and converts the ENTIRE content to a Markdown table for Annexure 1."""
    json_path = get_fundamental_json(ticker)
    if not json_path: return "_Data not available_"

    try:
        with open(json_path, 'r') as f: data = json.load(f)
        if not data: return "_Data empty_"
        
        df = pd.DataFrame(data)
        # Sort by date desc for readability in annexure
        if "Report_Date_Official" in df.columns:
            df = df.sort_values("Report_Date_Official", ascending=False)
            
        return df.to_markdown(index=False)
    except:
        return "_Error processing data_"

def compute_final_score(raw_scores: dict) -> dict:
    """Python-side calculation for deterministic scoring."""
    weighted = {k: raw_scores.get(k, 0) * SCORING_WEIGHTS[k] for k in SCORING_WEIGHTS}
    total = sum(weighted.values())
    max_score = 10 * sum(SCORING_WEIGHTS.values())
    pct = round((total / max_score) * 100, 1)

    if pct >= 70: rec = "Bullish"
    elif pct >= 50: rec = "Neutral"
    else: rec = "Bearish"

    return {
        "raw_scores": raw_scores, 
        "weighted_scores": weighted, 
        "total_score": total, 
        "percent": pct, 
        "recommendation": rec
    }

def extract_json_scores(text: str) -> dict:
    """Robust JSON extraction from LLM markdown response."""
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return json.loads(match.group(1))
        
        # Fallback
        match = re.search(r"(\{.*\"News\".*\})", text, re.DOTALL)
        if match: return json.loads(match.group(1))
    except: pass
    return {}

def format_score_card(score_data: dict) -> str:
    """Generates a Markdown Scorecard table."""
    if not score_data: return ""
    rec_emoji = "🐂" if score_data['recommendation'] == "Bullish" else "🐻" if score_data['recommendation'] == "Bearish" else "⚖️"
    
    md = f"### 🎯 AI Investment Scorecard\n| Category | Weight | Score (0-10) | Weighted |\n| :--- | :---: | :---: | :---: |\n"
    for k, weight in SCORING_WEIGHTS.items():
        raw = score_data['raw_scores'].get(k, 0)
        w = score_data['weighted_scores'].get(k, 0)
        md += f"| **{k}** | {weight}x | {raw} | {w} |\n"
    md += f"| **TOTAL** | | | **{score_data['total_score']} / {sum(SCORING_WEIGHTS.values())*10}** |\n"
    md += f"\n**Final Verdict: {score_data['percent']}%** {rec_emoji} **{score_data['recommendation']}**\n\n---\n"
    return md

def save_full_report(ticker, markdown_content, charts_filepath, annexure_1_markdown, annexure_2_markdown, annexure_3_markdown):
    """
    Saves the final report as HTML.
    Includes:
    1. Main Thesis & Scorecard
    2. Interactive Charts
    3. Annexure 1: Raw Fundamental Data (Table)
    4. Annexure 2: Market Research Data (Text)
    5. Annexure 3: Forensic Analysis (Text/Tables)
    """
    
    # 1. Read Charts
    charts_html = "<p><em>Charts not available.</em></p>"
    if charts_filepath and os.path.exists(charts_filepath):
        with open(charts_filepath, "r", encoding="utf-8") as f: charts_html = f.read()

    # 2. Convert Content to HTML
    body_html = markdown.markdown(markdown_content, extensions=['tables'])
    annexure_1_html = markdown.markdown(annexure_1_markdown, extensions=['tables'])
    annexure_2_html = markdown.markdown(annexure_2_markdown, extensions=['tables'])
    annexure_3_html = markdown.markdown(annexure_3_markdown, extensions=['tables'])
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Investment Memo: {ticker}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; background: #f9f9f9; color: #333; }}
            .report-container {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 30px; }}
            h3 {{ color: #7f8c8d; margin-top: 20px; }}
            
            /* Tables */
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; }}
            th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }} 
            th {{ background: #f2f2f2; font-weight: 600; }}
            tr:nth-child(even) {{ background-color: #fafafa; }}
            
            /* Visual Sections */
            .charts-container {{ margin-top: 40px; border-top: 3px solid #3498db; padding-top: 20px; }}
            
            /* Annexure 1 (Data Table) */
            .annexure-container {{ margin-top: 50px; border-top: 3px solid #e74c3c; padding-top: 20px; }}
            .annexure-container h2 {{ color: #c0392b; }}
            .annexure-table-wrapper {{ overflow-x: auto; }}
            
            /* Annexures 2 & 3 (Text Reports) */
            .annexure-text-wrapper {{ 
                background-color: #f8f9fa; 
                padding: 20px; 
                border-radius: 5px; 
                border: 1px solid #e9ecef;
                font-size: 0.95em;
                overflow-x: auto; /* Ensures wide forensic tables don't break layout */
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            {body_html}
            
            <div class="charts-container">
                <h2>Financial Visuals</h2>
                {charts_html}
            </div>
            
            <div class="annexure-container">
                <h2>Annexure 1: Raw Fundamental Data</h2>
                <p><em>Full historical dataset used for analysis.</em></p>
                <div class="annexure-table-wrapper">
                    {annexure_1_html}
                </div>
            </div>

            <div class="annexure-container">
                <h2>Annexure 2: Market Research & Sentiment</h2>
                <p><em>Raw data from Analyst Agent (News, Moat, Management).</em></p>
                <div class="annexure-text-wrapper">
                    {annexure_2_html}
                </div>
            </div>

            <div class="annexure-container">
                <h2>Annexure 3: Forensic Analysis</h2>
                <p><em>Detailed earnings quality, accounting checks, and financial health trends.</em></p>
                <div class="annexure-text-wrapper">
                    {annexure_3_html}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, "outputs", f"Investment_Memo_{ticker}.html")
    with open(output_path, "w", encoding="utf-8") as f: f.write(full_html)
    return output_path

# --- AGENT DEFINITIONS ---

def get_reporter_agent(model_id=MODEL_ID):
    return Agent(
        model=Ollama(id=model_id),
        description="You are a Pragmatic Portfolio Manager (GARP Focused).",
        instructions=[
            "Your goal is to write a balanced Investment Memo focusing on 'Growth at a Reasonable Price'.",
            "CRITICAL: Do not write a separate 'Risks' section. Interweave risks directly into your thesis.",
            
            # --- FUSED GROWTH THINKING ---
            "Since we are targetting growth at a reasonable premium, the focus should be whether the PE ratio plotted in the charts has room to grow the PE, given the current price, earnings, sentiment, market position and management.",
            "PE can grow in only two ways:",
            "  1) Growth in earnings with positive change in current stock price.",
            "  2) Market sentiment leads to rerating of stock to a higher PE.",
            "The reverse is also true (PE contraction) and you must flag the risk of a reversal too.",
            "While doing so, be balanced: neither a pessimist nor an optimist, but a pragmatist and realist.",
            
            "Output formatting: Use clear Markdown. Integers for scores."
        ],
        debug_mode=True,
    )

# --- WORKFLOW NODES (STEPS) ---

def ingest_node(state: MemoState):
    """Step 1: Gather all files and data."""
    analyst_file = get_latest_file(state.ticker, "research_")
    forensic_file = get_latest_file(state.ticker, "forensic_")
    
    state.market_data = get_file_content(analyst_file)
    state.forensic_data = get_file_content(forensic_file)
    
    # 1. Build Agent Context
    val_data = build_valuation_facts(state.ticker)
    if val_data["available"]:
        state.valuation_facts = val_data
        state.valuation_table = val_data["table_str"]
    else:
        state.valuation_table = "Valuation Data Unavailable."
        
    # 2. Build Full Annexure 1 (JSON Table)
    state.full_annexure_table = build_full_annexure(state.ticker)

def scoring_node(state: MemoState, agent):
    """Step 2: Pure quantitative scoring based on facts."""
    prompt = f"""
    Evaluate {state.ticker} based on these inputs.
    
    [MARKET DATA]: {state.market_data}
    [FORENSIC DATA]: {state.forensic_data}
    [VALUATION TABLE]:\n{state.valuation_table}
    
    Task: Assign scores (0-10) for these categories.
    Output ONLY a valid JSON object. No markdown, no text.
    {{
        "News": <int>,
        "Moat": <int>,
        "Management": <int>,
        "Earnings Quality": <int>
    }}
    """
    response = agent.run(prompt)
    state.scores = extract_json_scores(response.content)
    
    # Compute totals immediately
    if state.scores:
        state.scoring_data = compute_final_score(state.scores)
    else:
        # Default safety
        state.scoring_data = {"recommendation": "Hold", "total_score": 0, "percent": 0, "raw_scores": {}, "weighted_scores": {}}

def thesis_node(state: MemoState, agent):
    """Step 3: Write the narrative thesis."""
    # We pass the computed recommendation so the tone matches the score
    rec = state.scoring_data.get("recommendation", "Neutral")
    
    prompt = f"""
    Write the Core Thesis for {state.ticker}.
    
    [CONTEXT]
    The Quantitative Score is: {state.scoring_data.get('total_score')} ({rec}).
    
    [VALUATION FACTS - SOURCE OF TRUTH]
    {state.valuation_table}
    
    [RESEARCH INPUTS]
    {state.market_data[:3000]}
    
    Task:
    1. Write a 'Core Thesis' section. Focus on: Is the PE expansion sustainable given the data?
    2. Write a 'Forensic Financial Summary'. Use exact numbers from the Valuation Facts table.
    3. Write 'What Must Go Right / What Breaks'.
    4. INTERWEAVE risks. Use the Growth Thinking framework (PE expansion vs Contraction).
    """
    response = agent.run(prompt)
    state.thesis = response.content

def assembly_node(state: MemoState):
    """Step 4: Stitch it together."""
    scorecard = format_score_card(state.scoring_data)
    
    snapshot = f"""
# Investment Memo: {state.ticker}
    
## Investment Snapshot
* **Recommendation:** {state.scoring_data.get('recommendation')}
* **Score:** {state.scoring_data.get('percent')}%
* **Current P/E:** {state.valuation_facts.get('current_pe', 'N/A')}
* **Current PEG:** {state.valuation_facts.get('current_peg', 'N/A')}
    """
    
    state.final_markdown = f"{snapshot}\n\n{scorecard}\n\n{state.thesis}"

# --- MAIN ORCHESTRATOR ---

def generate_investment_memo(ticker):
    print(f"\n🏆 Reporter Agent: Assembling Investment Memo for {ticker}...")
    start_time = time.time()
    
    # Initialize State & Agents
    state = MemoState(ticker)
    reporter = get_reporter_agent()
    
    # Execution Pipeline
    with tqdm(total=4, desc="Workflow", unit="step") as pbar:
        
        # 1. Ingest
        pbar.set_description("Step 1/4: Ingesting Data")
        ingest_node(state)
        pbar.update(1)
        
        # 2. Score
        pbar.set_description("Step 2/4: Scoring Model")
        scoring_node(state, reporter)
        pbar.update(1)
        
        # 3. Thesis
        pbar.set_description("Step 3/4: Drafting Thesis")
        thesis_node(state, reporter)
        pbar.update(1)
        
        # 4. Assemble & Save
        pbar.set_description("Step 4/4: Final Assembly")
        assembly_node(state)
        
        chart_file = get_latest_file(ticker, "charts_")
        
        # ✅ Updated Call: Passing Annexure 1, 2, AND 3
        output_path = save_full_report(
            ticker, 
            state.final_markdown, 
            chart_file, 
            state.full_annexure_table, 
            state.market_data,
            state.forensic_data
        )
        pbar.update(1)

    end_time = time.time()
    formatted_time = str(timedelta(seconds=int(end_time - start_time)))
    print(f"\n⏱️ Total Execution Time: {formatted_time}")
    print(f"✅ Investment Memo Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    ticker = input("Enter ticker: ").strip()
    generate_investment_memo(ticker)