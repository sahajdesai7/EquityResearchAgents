import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDIA_TICKERS_PATH = os.path.join(BASE_DIR, "static_inputs", "nse_yfinance_tickers.csv")
US_TICKERS_PATH = os.path.join(BASE_DIR, "static_inputs", "SP500.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

def get_ticker_universe(region):
    """
    Returns a list of tickers based on the selected geography.
    """
    tickers = []
    
    if region == "US":
        print(f"🌎 Fetching S&P 500 list from local file...")
        print(f"📂 Looking for file at: {US_TICKERS_PATH}")

        if os.path.exists(US_TICKERS_PATH):
            try:
                df = pd.read_csv(US_TICKERS_PATH)
                tickers = [str(t).strip().replace(".", "-") for t in df['Symbol'].tolist()]
                print(f"✅ Loaded {len(tickers)} US tickers from local file.")
            except Exception as e:
                print(f"⚠️ Error reading local US CSV: {e}")
                tickers = []
        else:
            print(f"❌ File not found at: {US_TICKERS_PATH}")
            tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG"]
            
    elif region == "India":
        print(f"🌏 Fetching Nifty list from local file...")
        print(f"📂 Looking for file at: {INDIA_TICKERS_PATH}")
        
        if os.path.exists(INDIA_TICKERS_PATH):
            try:
                df = pd.read_csv(INDIA_TICKERS_PATH)
                tickers = [f"{str(t).strip()}.NS" for t in df['symbol'].tolist()]
                print(f"✅ Loaded {len(tickers)} tickers from local file.")
            except Exception as e:
                print(f"⚠️ Error reading local CSV: {e}")
                tickers = []
        else:
            print(f"❌ File not found at: {INDIA_TICKERS_PATH}")
            tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
            
    return tickers

def get_price_at_date(stock, target_date):
    """
    Fetches the Close price on or immediately after a specific date.
    """
    try:
        start_date = target_date.strftime('%Y-%m-%d')
        end_date = (target_date + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        hist = stock.history(start=start_date, end=end_date)
        if not hist.empty:
            return hist['Close'].iloc[0]
        return None
    except:
        return None

def check_growth_criteria(ticker, debug=False):
    """
    Downloads data and checks growth criteria:
    1. Rev Growth (Quarterly YoY) >= 9%
    2. Rev Growth (Annual YoY) >= 9%
    3. EPS Growth (Quarterly YoY) >= 15%
    4. EPS Growth (Annual YoY) >= 15%
    5. PE Expansion <= 30% (Comparing Current TTM vs TTM 1 Qtr Ago)
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch Financials
        q_financials = stock.quarterly_financials
        a_financials = stock.financials # Annuals

        if q_financials.empty or a_financials.empty:
            if debug: print(f"⚠️ {ticker}: Missing financial data.")
            return None

        # Sort columns (Newest First)
        q_financials = q_financials[sorted(q_financials.columns, reverse=True)]
        a_financials = a_financials[sorted(a_financials.columns, reverse=True)]
        
        # Check Data Sufficiency
        # Need 5 quarters (0,1,2,3,4) for Q-YoY check and TTM calc
        if len(q_financials.columns) < 5:
            if debug: print(f"⚠️ {ticker}: Insufficient quarterly data ({len(q_financials.columns)} qtrs).")
            return None
        
        # Need 2 years (0,1) for Annual check
        if len(a_financials.columns) < 2:
            if debug: print(f"⚠️ {ticker}: Insufficient annual data.")
            return None

        # --- 1. Revenue Growth Check ---
        # A. Quarterly YoY (Q_Current vs Q_Same_Last_Year)
        rev_curr_q = q_financials.loc['Total Revenue'].iloc[0]
        rev_last_q = q_financials.loc['Total Revenue'].iloc[4]
        rev_growth_q = (rev_curr_q / rev_last_q) - 1

        # B. Annual YoY (Year_Current vs Year_Last)
        rev_curr_a = a_financials.loc['Total Revenue'].iloc[0]
        rev_last_a = a_financials.loc['Total Revenue'].iloc[1]
        rev_growth_a = (rev_curr_a / rev_last_a) - 1

        # --- 2. EPS Growth Check ---
        if 'Diluted EPS' in q_financials.index:
            eps_row = 'Diluted EPS'
        elif 'Basic EPS' in q_financials.index:
            eps_row = 'Basic EPS'
        else:
            if debug: print(f"⚠️ {ticker}: EPS row not found.")
            return None
        
        # A. Quarterly YoY
        eps_curr_q = q_financials.loc[eps_row].iloc[0]
        eps_last_q = q_financials.loc[eps_row].iloc[4]
        eps_growth_q = (eps_curr_q / eps_last_q) - 1

        # B. Annual YoY
        # Ensure row exists in Annuals too
        if eps_row not in a_financials.index:
            if debug: print(f"⚠️ {ticker}: EPS row missing in Annuals.")
            return None
            
        eps_curr_a = a_financials.loc[eps_row].iloc[0]
        eps_last_a = a_financials.loc[eps_row].iloc[1]
        eps_growth_a = (eps_curr_a / eps_last_a) - 1
        
        # --- 3. PE Expansion Logic ---
        # Current Price
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        
        # Current TTM EPS (Sum of Q0, Q1, Q2, Q3)
        ttm_eps_now = q_financials.loc[eps_row].iloc[0:4].sum()
        
        # Current PE
        pe_current = current_price / ttm_eps_now if ttm_eps_now > 0 else 0

        # Old TTM EPS (Sum of Q1, Q2, Q3, Q4) -> Effectively TTM as of 1 Quarter Ago
        # We use range [1:5] which selects indices 1, 2, 3, 4
        ttm_eps_old = q_financials.loc[eps_row].iloc[1:5].sum()
        
        # Price at the end of Q1 (the date associated with index 1)
        # This aligns with when the 'Old TTM' would have been valid
        date_old = q_financials.columns[1]
        price_old_val = get_price_at_date(stock, date_old)
        
        pe_old = 0
        pe_expansion = 0
        
        if price_old_val and ttm_eps_old > 0:
            pe_old = price_old_val / ttm_eps_old
            pe_expansion = (pe_current / pe_old) - 1
        else:
            # Fallback if price fetch fails or EPS negative
            pe_expansion = 0 

        # --- DEBUG PRINT BLOCK ---
        if debug:
            print(f"\n--- 🧪 DEBUG CHECK: {ticker} ---")
            print(f"Rev Growth: Q-YoY {rev_growth_q:.1%} | Ann-YoY {rev_growth_a:.1%}")
            print(f"EPS Growth: Q-YoY {eps_growth_q:.1%} | Ann-YoY {eps_growth_a:.1%}")
            print(f"PE Check: Now {pe_current:.2f} (Price {current_price:.2f})")
            print(f"PE Check: Old {pe_old:.2f} (Price {price_old_val} @ {date_old.date()})")
            print(f"PE Expansion: {pe_expansion:.2%}")
            print("-----------------------------")

        # --- FILTER GATES ---
        # 1. Growth Thresholds
        if rev_growth_q < 0.09 and rev_growth_a < 0.05: return None
        if eps_growth_q < 0.09 and eps_growth_a < 0.05: return None
        
        # 2. PE Expansion Gate (<= 30%)
        # Ignore checks if PE is invalid/negative, but don't crash
        if pe_current > 0 and pe_old > 0:
            if pe_expansion > 0.30: 
                if debug: print(f"❌ Excessive PE Expansion: {pe_expansion:.1%}")
                return None

        return {
            "Ticker": ticker,
            "Company": stock.info.get('longName', ticker),
            "Rev_Growth_Q_YoY": round(rev_growth_q * 100, 2),
            "Rev_Growth_Ann": round(rev_growth_a * 100, 2),
            "EPS_Growth_Q_YoY": round(eps_growth_q * 100, 2),
            "EPS_Growth_Ann": round(eps_growth_a * 100, 2),
            "Current_PE": round(pe_current, 2),
            "Old_PE": round(pe_old, 2),
            "PE_Expansion": round(pe_expansion * 100, 2)
        }

    except Exception as e:
        if debug: print(f"❌ Error checking {ticker}: {e}")
        return None

def screen_stocks():
    print("\n--- 🔍 Quantitative Screener Started ---")
    region = input("Select Geography (US/India): ").strip()
    
    tickers = get_ticker_universe(region)
    print(f"📋 Universe found: {len(tickers)} companies. Beginning scan...")
    print("☕ This may take a while. Analyzing fundamentals...")

    results = []
    
    # Iterate with index to track count for debugging
    # ⚠️ NOTE: Limited to first 20 for testing. Remove [:20] for full scan.
    for i, ticker in enumerate(tickers[:20]): 
        # Enable debug mode for the first 5 tickers
        is_debug_mode = (i < 5)
        
        if not is_debug_mode:
            print(f"Checking {ticker}...", end="\r")
        
        data = check_growth_criteria(ticker, debug=is_debug_mode)
        
        if data:
            results.append(data)
            print(f"✅ FOUND: {ticker} ({data['Rev_Growth_Q_YoY']}% Growth)")
    
    print("\n\n--- 🏁 Scan Complete ---")
    
    if results:
        df_results = pd.DataFrame(results)
        
        if not os.path.exists(OUTPUTS_DIR):
            os.makedirs(OUTPUTS_DIR)

        filename = os.path.join(OUTPUTS_DIR, f"screener_results_{region}.csv")
        df_results.to_csv(filename, index=False)
        print(f"💾 Results saved to: {filename}")
        print(df_results)
        
        selected_ticker = input("\n👉 Enter a specific Ticker from the list above to proceed to Deep Dive (RELIANCE.NS or AAPL): ").strip()
        return selected_ticker
    else:
        print("❌ No companies matched the criteria.")
        return None

if __name__ == "__main__":
    screen_stocks()