import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np

# Agno (Phidata) Imports
from agno.agent import Agent
from agno.models.ollama import Ollama 
from ddgs import DDGS 
from tqdm import tqdm 

# Load environment variables
load_dotenv()

# --- Configuration ---
OLLAMA_MODEL_ID = "llama3.2:3b"

# Define Output Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# --- Helper Functions: Quantitative ---

def get_currency_rate(base_currency="USD", target_currency="INR"):
    """
    Fetches the current exchange rate.
    """
    if base_currency == target_currency:
        return 1.0
    try:
        pair = f"{target_currency}=X" 
        data = yf.Ticker(pair)
        rate = data.history(period="1d")['Close'].iloc[-1]
        return round(rate, 2)
    except Exception:
        return 1.0

def process_financials(financials, balance_sheet, period_name="Annual"):
    """
    Internal helper to process a set of financial statements into a clean DataFrame.
    """
    if financials.empty or balance_sheet.empty:
        return pd.DataFrame()

    # Align Data (Transpose and Merge)
    fin_T = financials.T
    bs_T = balance_sheet.T
    
    # Inner join on Date index to align periods
    combined = pd.merge(fin_T, bs_T, left_index=True, right_index=True, how='inner')
    
    # Sort by date descending (newest first) and take relevant rows
    limit = 3 if period_name == "Annual" else 5
    recent_history = combined.sort_index(ascending=False).head(limit)
    
    data = []
    for date, row in recent_history.iterrows():
        # --- A. Revenue & Income ---
        revenue = row.get('Total Revenue', row.get('Revenue', 0))
        net_income = row.get('Net Income', 0)
        
        # --- B. Balance Sheet Items ---
        equity = row.get('Stockholders Equity', row.get('Total Stockholder Equity', 1))
        total_debt = row.get('Total Debt', 0)
        
        # --- C. Ratios ---
        ronw = (net_income / equity) * 100 if equity != 0 else 0
        debt_to_equity = total_debt / equity if equity != 0 else 0
        net_margin = (net_income / revenue) * 100 if revenue != 0 else 0
        
        # --- D. Interest Coverage ---
        interest_exp = abs(row.get('Interest Expense', 0))
        # Fallback for EBIT
        ebit = row.get('EBIT', row.get('Pretax Income', 0) + interest_exp)
        
        if interest_exp == 0:
            int_coverage = 999.0 
        else:
            int_coverage = ebit / interest_exp
            
        data.append({
            "Period": date.strftime('%Y-%m-%d'),
            "Type": period_name,
            "Revenue": f"{revenue:,.0f}",
            "Net Margin (%)": round(net_margin, 2),
            "RONW (%)": round(ronw, 2),
            "Debt/Equity": round(debt_to_equity, 2),
            "Int. Coverage": round(int_coverage, 2),
            "Net Income": f"{net_income:,.0f}",
            "EBIT": f"{ebit:,.0f}",
            "Total Debt": f"{total_debt:,.0f}"
        })
    
    return pd.DataFrame(data)

def calculate_ttm(ticker):
    """
    Calculates Trailing Twelve Months (TTM) data.
    Logic: Sums flows (Income/Revenue) over last 4 quarters, takes latest snapshot for Balance Sheet.
    """
    q_fin = ticker.quarterly_financials
    q_bs = ticker.quarterly_balance_sheet
    
    if q_fin.empty or q_bs.empty:
        return pd.DataFrame()

    # Ensure descending order (Newest -> Oldest)
    q_fin = q_fin.sort_index(axis=1, ascending=False)
    q_bs = q_bs.sort_index(axis=1, ascending=False)

    # We need at least 4 quarters for TTM
    if len(q_fin.columns) < 4:
        return pd.DataFrame()

    # 1. Sum Flows (Last 4 Quarters)
    last_4 = q_fin.iloc[:, :4]
    
    revenue_ttm = last_4.loc['Total Revenue'].sum() if 'Total Revenue' in last_4.index else 0
    net_income_ttm = last_4.loc['Net Income'].sum() if 'Net Income' in last_4.index else 0
    
    interest_series = last_4.loc['Interest Expense'] if 'Interest Expense' in last_4.index else pd.Series([0])
    interest_exp_ttm = abs(interest_series.sum())
    
    if 'EBIT' in last_4.index:
        ebit_ttm = last_4.loc['EBIT'].sum()
    elif 'Pretax Income' in last_4.index:
        ebit_ttm = last_4.loc['Pretax Income'].sum() + interest_exp_ttm
    else:
        ebit_ttm = 0

    # 2. Latest Snapshot (Most Recent Quarter) for Balance Sheet
    latest_bs = q_bs.iloc[:, 0]
    equity = latest_bs.get('Stockholders Equity', latest_bs.get('Total Stockholder Equity', 1))
    total_debt = latest_bs.get('Total Debt', 0)

    # 3. Calculate Ratios
    ronw = (net_income_ttm / equity) * 100 if equity != 0 else 0
    debt_to_equity = total_debt / equity if equity != 0 else 0
    net_margin = (net_income_ttm / revenue_ttm) * 100 if revenue_ttm != 0 else 0
    int_coverage = 999.0 if interest_exp_ttm == 0 else ebit_ttm / interest_exp_ttm

    data = [{
        "Period": "TTM (Last 4Q)",
        "Type": "Trailing 12M",
        "Revenue": f"{revenue_ttm:,.0f}",
        "Net Margin (%)": round(net_margin, 2),
        "RONW (%)": round(ronw, 2),
        "Debt/Equity": round(debt_to_equity, 2),
        "Int. Coverage": round(int_coverage, 2),
        "Net Income": f"{net_income_ttm:,.0f}",
        "EBIT": f"{ebit_ttm:,.0f}",
        "Total Debt": f"{total_debt:,.0f}"
    }]
    return pd.DataFrame(data)

def enrich_json_data(ticker):
    """
    Reads the existing fundamentals JSON (created by charts.py) and appends
    Forensic metrics. Falls back to Annual Data if Quarterly is missing.
    """
    print(f"   🧬 Enriching JSON with Forensic Data (TTM + Annual Fallback)...")
    
    safe_ticker = ticker.replace(".", "_")
    filename = f"fundamentals_{ticker}.json"
    filepath = os.path.join(OUTPUTS_DIR, filename)

    if not os.path.exists(filepath):
        print("   ⚠️ JSON file not found. Skipping enrichment.")
        return

    # 1. Load Existing Data
    with open(filepath, 'r') as f:
        json_data = json.load(f)
        
    # 2. Fetch Full History (Quarterly AND Annual)
    stock = yf.Ticker(ticker)
    
    # Quarterly (Recent detailed view)
    q_fin = stock.quarterly_financials.T
    q_bs = stock.quarterly_balance_sheet.T
    
    # Annual (Longer history fallback)
    a_fin = stock.financials.T
    a_bs = stock.balance_sheet.T
    
    # Clean indices
    for df in [q_fin, q_bs, a_fin, a_bs]:
        if df.index.tz is not None: df.index = df.index.tz_localize(None)

    # 3. Iterate and Enrich
    for record in json_data:
        report_date_str = record.get("Report_Date_Official")
        if not report_date_str: continue
        
        report_date = pd.Timestamp(report_date_str)
        
        # Init Variables
        revenue = None
        net_income = None
        equity = None
        total_debt = None
        ebit = None
        interest_exp = None
        
        data_source = "None"

        # --- STRATEGY A: Try Quarterly TTM First (Preferred) ---
        if report_date in q_fin.index and report_date in q_bs.index:
            try:
                # Find integer location
                q_fin_sorted = q_fin.sort_index(ascending=False)
                loc_sorted = q_fin_sorted.index.get_loc(report_date)
                
                # We need 4 quarters (current + 3 past)
                if loc_sorted + 4 <= len(q_fin_sorted):
                    revenue = q_fin_sorted['Total Revenue'].iloc[loc_sorted : loc_sorted+4].sum()
                    net_income = q_fin_sorted['Net Income'].iloc[loc_sorted : loc_sorted+4].sum()
                    
                    # Interest & EBIT
                    int_series = q_fin_sorted.get('Interest Expense', pd.Series(0))
                    interest_exp = abs(int_series.iloc[loc_sorted : loc_sorted+4].sum())

                    if 'EBIT' in q_fin_sorted.columns:
                        ebit = q_fin_sorted['EBIT'].iloc[loc_sorted : loc_sorted+4].sum()
                    elif 'Pretax Income' in q_fin_sorted.columns:
                        ebit = q_fin_sorted['Pretax Income'].iloc[loc_sorted : loc_sorted+4].sum() + interest_exp
                    else:
                        ebit = 0
                    
                    # Balance Sheet (Snapshot)
                    row_bs = q_bs.loc[report_date]
                    equity = row_bs.get('Stockholders Equity', row_bs.get('Total Stockholder Equity', 1))
                    total_debt = row_bs.get('Total Debt', 0)
                    
                    data_source = "Quarterly_TTM"
            except Exception:
                pass

        # --- STRATEGY B: Fallback to Annual (If TTM failed) ---
        if data_source == "None" and report_date in a_fin.index and report_date in a_bs.index:
            try:
                row_fin = a_fin.loc[report_date]
                row_bs = a_bs.loc[report_date]
                
                revenue = row_fin.get('Total Revenue', 0)
                net_income = row_fin.get('Net Income', 0)
                
                interest_exp = abs(row_fin.get('Interest Expense', 0))
                
                if 'EBIT' in row_fin:
                    ebit = row_fin['EBIT']
                elif 'Pretax Income' in row_fin:
                    ebit = row_fin['Pretax Income'] + interest_exp
                else:
                    ebit = 0
                    
                equity = row_bs.get('Stockholders Equity', row_bs.get('Total Stockholder Equity', 1))
                total_debt = row_bs.get('Total Debt', 0)
                
                data_source = "Annual"
            except Exception:
                pass

        # --- CALCULATE & SAVE RATIOS ---
        # Only proceed if we found valid data from either source
        if revenue is not None and equity is not None:
            # 1. Net Margin
            net_margin = (net_income / revenue) * 100 if revenue != 0 else 0
            
            # 2. RONW (ROE)
            ronw = (net_income / equity) * 100 if equity != 0 else 0
            
            # 3. Debt/Equity
            de_ratio = total_debt / equity if equity != 0 else 0
            
            # 4. Interest Coverage
            int_cov = ebit / interest_exp if interest_exp > 0 else 999.0

            # Append to Record
            record['Data_Source_Type'] = data_source
            record['Revenue'] = round(revenue, 2)
            record['Net_Income'] = round(net_income, 2)
            record['EBIT'] = round(ebit, 2)
            record['Total_Debt'] = round(total_debt, 2)
            record['Equity'] = round(equity, 2)
            
            record['Net_Margin_Pct'] = round(net_margin, 2)
            record['RONW_Pct'] = round(ronw, 2)
            record['Debt_to_Equity'] = round(de_ratio, 2)
            record['Interest_Coverage'] = round(int_cov, 2)
            
        else:
            # Explicitly mark missing data for clarity
            record['Data_Source_Type'] = "Missing"

    # 4. Save Updates
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"   ✅ JSON enriched with Revenue, Debt, EBIT & Ratios (Coverage: {len(json_data)} points).")

def get_historical_ratios(ticker_symbol):
    """Fetches Annual, Quarterly, and TTM history."""
    ticker = yf.Ticker(ticker_symbol)
    
    # Metadata
    try:
        info = ticker.info
        metadata = {
            "currency": info.get('currency', 'USD'),
            "long_name": info.get('longName', ticker_symbol)
        }
    except:
        metadata = {"currency": "USD", "long_name": ticker_symbol}

    # 1. Annual & Quarterly
    annual_df = process_financials(ticker.financials, ticker.balance_sheet, "Annual")
    quarterly_df = process_financials(ticker.quarterly_financials, ticker.quarterly_balance_sheet, "Quarter")
    
    # 2. TTM Data (New)
    ttm_df = calculate_ttm(ticker)
    
    return annual_df, quarterly_df, ttm_df, metadata

# --- Helper Functions: Search ---

def search_duckduckgo(query: str, max_results=5) -> str:
    """Executes the search via Python."""
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return f"No results found for: {query}"
        
        output = []
        for article in results:
            title = article.get('title', 'N/A')
            body = article.get('body', 'N/A')
            url = article.get('href', 'N/A')
            output.append(f"Title: {title}\nSummary: {body}\nLink: {url}\n---")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching DuckDuckGo: {e}"

def get_forensic_agent():
    return Agent(
        model=Ollama(id=OLLAMA_MODEL_ID), 
        description="You are a strict forensic accountant. You are not a pessimist but a pragmatist.",
        instructions=[
            # Core Mission
            "Your job is to identify low-quality earnings and deteriorating trends.",
            "Compare the TTM (Current) vs Annual (History) vs Quarterly (Recent Trend).",
            
            # --- NEW: PRECISION GUARDRAILS ---
            "Verify 'Exceptional Items': actively search for non-recurring gains/losses (restructuring, fines, asset sales) that distort the P/E ratio.",
            "MATERIALITY TEST: Ignore minor operational charges. Only flag items that significantly alter the Net Margin or EPS trajectory.",
            "Use the quantitative data (RONW, Debt/Equity) as the baseline truth; if search results contradict the hard data, prioritize the data but note the discrepancy.",
            # ---------------------------------
            
            "Output concise observations. Do not summarize; analyze.",
        ],
        debug_mode=True,
        markdown=True,
    )

def safe_run_agent(agent, prompt, annual_df, quarterly_df, ttm_df, metadata, sleep_time=1):
    """Cycle: Search (Python) -> Analyze (Agent)."""
    company = metadata.get('long_name', 'The Company')
    
    # --- UPGRADE: Multi-Year Search Strategy ---
    current_year = datetime.now().year
    years_to_search = [current_year, current_year - 1, current_year - 2]
    
    accumulated_search_results = []
    
    tqdm.write(f"   🔎 Searching for exceptional items over {len(years_to_search)} years...")
    
    for year in years_to_search:
        # We search specifically for that year's exceptional items
        search_query = f"{company} exceptional items one-time charges write-off financial results {year}"
        # tqdm.write(f"      - Scanning {year}...") # Optional detail log
        
        # Get results for this specific year
        year_results = search_duckduckgo(search_query)
        
        # Add a header so the Agent knows which year this text belongs to
        accumulated_search_results.append(f"--- SEARCH RESULTS FOR {year} ---\n{year_results}")
        
        # Be polite to the search API
        time.sleep(1) 

    # Combine all results into one big context string
    full_search_context = "\n\n".join(accumulated_search_results)
    
    # Convert DataFrames to Markdown
    ttm_md = ttm_df.to_markdown(index=False) if not ttm_df.empty else "No TTM Data"
    annual_md = annual_df.to_markdown(index=False) if not annual_df.empty else "No Annual Data"
    quarterly_md = quarterly_df.to_markdown(index=False) if not quarterly_df.empty else "No Quarterly Data"
    
    final_prompt = f"""
    {prompt}

    === TTM SNAPSHOT (MOST RECENT 12 MONTHS) ===
    {ttm_md}

    === 3-YEAR ANNUAL HISTORY ({metadata.get('currency')}) ===
    {annual_md}
    
    === RECENT QUARTERLY TRENDS ===
    {quarterly_md}
    
    * Benchmarks:
      - RONW: >14% (Emerging), >7% (Developed)
      - Debt/Equity: <1.5 Preferred
      - Interest Coverage: >3.0 Safe

    === QUALITATIVE SEARCH RESULTS (3-YEAR HISTORY) ===
    {full_search_context}
    ===================================================

    **Task**:
    1. **Current Health**: Look at TTM figures. Are they healthy?
    2. **Trend Analysis**: Compare TTM vs previous Years. Is the situation improving or deteriorating?
    3. **Quality Check**: Scan the search results for EACH year. Do you see specific one-time gains/losses that distort the trend for that specific year?
    4. **Verdict**: Is the company fundamentally healthy?
    """
    
    for _ in range(2):
        try:
            time.sleep(sleep_time)
            response = agent.run(final_prompt)
            if response and response.content:
                return response.content
        except Exception as e:
            tqdm.write(f"❌ Error in forensic agent: {e}")
            
    return "Analysis Failed."

def analyze_earnings_quality(ticker):
    """Main Workflow."""
    print(f"\n🧪 Forensic Agent ({OLLAMA_MODEL_ID}): Analyzing Earnings Quality for {ticker}...")
    
    # 1. Get History, TTM & Metadata
    annual_df, quarterly_df, ttm_df, metadata = get_historical_ratios(ticker)
    
    if annual_df.empty and quarterly_df.empty:
        print("❌ Could not fetch financial history. Aborting.")
        return None

    # 2. Get Exchange Rate
    if metadata['currency'] != 'USD':
        usd_rate = get_currency_rate("USD", metadata['currency'])
        metadata['usd_rate'] = usd_rate
        print(f"   💱 Exchange Rate: 1 USD = {usd_rate} {metadata['currency']}")
    else:
        metadata['usd_rate'] = 1.0

    print(f"   📊 Retrieved History + TTM Data.")

    # 3. Agentic Analysis
    agent = get_forensic_agent()
    
    with tqdm(total=2, desc="Forensic Scan", unit="step") as pbar:
        # Step A: Run the LLM Analysis
        pbar.set_description("Step 1/2: LLM Analysis")
        report_content = safe_run_agent(
            agent,
            prompt="Analyze the Earnings Quality and Financial Trends.",
            annual_df=annual_df,
            quarterly_df=quarterly_df,
            ttm_df=ttm_df,
            metadata=metadata
        )
        pbar.update(1)

        # Step B: Enrich the JSON data
        pbar.set_description("Step 2/2: Updating JSON Data")
        enrich_json_data(ticker)
        pbar.update(1)

    # 4. Save to File
    print(f"   💾 Saving Forensic Report...")
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    safe_ticker = ticker.replace(".", "_")
    filename = f"forensic_{safe_ticker}_{datetime.now().strftime('%Y%m%d')}.txt"
    filepath = os.path.join(OUTPUTS_DIR, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"FORENSIC EARNINGS REPORT FOR {metadata['long_name']} ({ticker})\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("[TRAILING 12 MONTHS (TTM)]\n")
            f.write(ttm_df.to_string(index=False))
            f.write("\n\n")

            f.write("[ANNUAL FINANCIAL TRENDS]\n")
            f.write(annual_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("[QUARTERLY FINANCIAL TRENDS]\n")
            f.write(quarterly_df.to_string(index=False))
            f.write("\n\n" + "-"*30 + "\n\n")
            
            f.write(report_content)
            f.write("\n" + "="*60 + "\n")
            
        print(f"✅ Forensic Report saved: {filepath}")
        return filepath

    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None

# --- Testing Block ---
if __name__ == "__main__":
    ticker = input("Enter ticker (e.g. NVDA or RELIANCE.NS): ").strip()
    analyze_earnings_quality(ticker)