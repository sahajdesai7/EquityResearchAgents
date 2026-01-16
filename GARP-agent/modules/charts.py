import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def print_debug_view(df, name):
    print(f"\n--- 🐞 DEBUG VIEW: {name} (Quarter Boundaries) ---")
    if df is None or df.empty:
        print("   [Empty Dataframe]")
        return

    current_year = datetime.now().year
    years = range(current_year - 5, current_year + 1)
    q_months = [3, 6, 9, 12]
    
    target_dates = []
    for y in years:
        for m in q_months:
            if m in [3, 12]: d = 31
            elif m in [6, 9]: d = 30
            q_date = datetime(y, m, d)
            target_dates.append(q_date)

    mask = pd.Series(False, index=df.index)
    for t_date in target_dates:
        start_w = t_date - timedelta(days=1)
        end_w = t_date + timedelta(days=1)
        mask |= (df.index >= start_w) & (df.index <= end_w)

    debug_df = df[mask]
    
    if debug_df.empty:
        print("   [No data found exactly around quarter ends]")
    else:
        print(debug_df.head(15))
        if len(debug_df) > 15:
            print(f"   ... and {len(debug_df) - 15} more rows.")
    print("--------------------------------------------------\n")


def get_fundamental_data(ticker):
    print(f"\n--- 🛠️ Processing Data for {ticker} ---")
    stock = yf.Ticker(ticker)

    # --- Step 2: Daily Closing Prices ---
    print("Step 2: Fetching Price History...")
    df_price = stock.history(period="5y")
    if df_price.empty: return None
    
    if df_price.index.tz is not None:
        df_price.index = df_price.index.tz_localize(None)
    
    df_price = df_price[['Close']].copy()
    print_debug_view(df_price, "Step 2: Price Dataframe")

    # --- Step 3 & 4: Fetch EPS ---
    print("Step 3 & 4: Fetching EPS Data...")
    annuals = stock.financials.T
    if annuals.index.tz is not None: annuals.index = annuals.index.tz_localize(None)
    
    quarters = stock.quarterly_financials.T
    if quarters.index.tz is not None: quarters.index = quarters.index.tz_localize(None)
    
    # 🔍 PRINT RAW DATA
    print("\n>>> RAW ANNUALS HEAD:")
    print(annuals['Diluted EPS'])
    print("\n>>> RAW QUARTERLY HEAD:")
    print(quarters['Diluted EPS'])

    eps_col = None
    for col in ['Diluted EPS', 'Basic EPS']:
        if col in annuals.columns and col in quarters.columns:
            eps_col = col
            break
            
    if not eps_col:
        print(f"⚠️ Could not find EPS column.")
        return None

    # --- Step 5: Calculate TTM EPS ---
    print("\nStep 5: Calculating TTM EPS...")
    
    # A. Annuals (Baseline History)
    series_annual_ttm = annuals[eps_col].dropna()

    # B. Quarterlies (Rolling Calculation)
    # Ensure sorted Oldest -> Newest so rolling works forward in time
    quarters.sort_index(inplace=True)
    
    # Logic: We need 4 quarters to make 1 TTM point.
    # If we have N quarters, we get (N - 3) valid TTM points.
    series_quarterly_ttm = quarters[eps_col].rolling(window=4).sum()
    print(series_quarterly_ttm)
    
    # Drop the NaNs created by the rolling window (the first 3 rows)
    series_quarterly_ttm = series_quarterly_ttm.dropna()
    print(series_annual_ttm)
    
    print(f"   -> Calculated {len(series_quarterly_ttm)} valid Quarterly TTM points from {len(quarters)} available quarters.")

    # --- Step 6: Combine Unified Table ---
    s_combined = pd.concat([series_annual_ttm, series_quarterly_ttm])
    s_combined.sort_index(inplace=True)
    
    # Handle overlaps: If an Annual date and Quarterly date match, keep Quarterly (more precise/recent)
    s_combined = s_combined[~s_combined.index.duplicated(keep='last')]
    
    df_eps = s_combined.to_frame(name='TTM_EPS')
    print_debug_view(df_eps, "Step 6: Combined EPS Dataframe")

    # --- Step 7: Merge and Fill ---
    print("Step 7: Merging and Filling...")
    
    # Forward fill logic: TTM EPS stays valid until the next report comes out
    df_eps_daily = df_eps.reindex(df_price.index, method='ffill')
    df_final = df_price.join(df_eps_daily)
    
    # --- Step 8: Calculate PE ---
    print("Step 8: Calculating PE Ratio...")
    df_final['PE_Ratio'] = df_final['Close'] / df_final['TTM_EPS']
    
    # Clean up artifacts
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.dropna(subset=['PE_Ratio'], inplace=True)
    
    print_debug_view(df_final, "Step 8: Final Merged Dataframe")
    
    return df_final

def create_charts(ticker):
    df = get_fundamental_data(ticker)
    if df is None or len(df) < 50: return None, None

    print("Step 9: Plotting...")
    
    # Chart A: Quantamental
    mean_pe = df['PE_Ratio'].mean()
    std_pe = df['PE_Ratio'].std()
    
    fig_fund = make_subplots(specs=[[{"secondary_y": True}]])
    fig_fund.add_trace(go.Scatter(x=df.index, y=df['TTM_EPS'], name="TTM EPS", fill='tozeroy', line=dict(width=0), fillcolor='rgba(46, 204, 113, 0.2)'), secondary_y=False)
    fig_fund.add_trace(go.Scatter(x=df.index, y=df['PE_Ratio'], name="PE Ratio", line=dict(color='#2980b9', width=2)), secondary_y=True)
    
    fig_fund.add_hline(y=mean_pe, line_dash="dot", line_color="black", secondary_y=True)
    fig_fund.add_hline(y=mean_pe + std_pe, line_dash="dash", line_color="orange", secondary_y=True)
    fig_fund.add_hline(y=mean_pe - std_pe, line_dash="dash", line_color="green", secondary_y=True)
    fig_fund.add_hline(y=mean_pe + 2*std_pe, line_dash="solid", line_color="red", secondary_y=True)

    fig_fund.update_layout(title=f"<b>{ticker} Quantamental Chart</b>", template="plotly_white", height=500)

    # Chart B: Technicals
    df['RSI'] = calculate_rsi(df['Close'])
    fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='black')), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
    fig_tech.add_hrect(y0=40, y1=60, row=2, col=1, fillcolor="blue", opacity=0.1, line_width=0)
    
    fig_tech.update_layout(title=f"<b>{ticker} Technicals</b>", template="plotly_white", height=600, showlegend=False)

    return fig_fund, fig_tech

def save_charts_to_html(fig1, fig2, filename="charts_TEST.html"):
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
        
    filepath = os.path.join(OUTPUTS_DIR, filename)
    with open(filepath, 'w') as f:
        f.write("<html><head><title>Quantamental Charts</title></head><body>")
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<br><hr><br>")
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</body></html>")
    print(f"\n✅ Charts saved to: {filepath}")

if __name__ == "__main__":
    ticker = input("Enter ticker (e.g., RELIANCE.NS or AAPL): ").strip()
    f1, f2 = create_charts(ticker)
    if f1 and f2:
        save_charts_to_html(f1, f2, f"charts_{ticker}.html")
        f1.show()
        f2.show()