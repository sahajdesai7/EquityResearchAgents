import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
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

    # Check for near-quarter-end dates for visibility
    mask = df.index.is_quarter_end
    debug_df = df[mask]
    if debug_df.empty:
        print("   [No exact quarter-end dates found in index. Showing tail:]")
        print(df.tail(5))
    else:
        print(debug_df.tail(10))
    print("--------------------------------------------------\n")

def get_fundamental_data(ticker):
    print(f"\n--- 🛠️ Processing Data for {ticker} ---")
    stock = yf.Ticker(ticker)

    # --- Step 2: Daily Closing Prices ---
    print("Step 2: Fetching Price History...")
    df_price = stock.history(period="5y")
    if df_price.empty: return None, []
    
    if df_price.index.tz is not None:
        df_price.index = df_price.index.tz_localize(None)
    
    df_price = df_price[['Close']].copy()

    # --- Step 3 & 4: Fetch EPS ---
    print("Step 3 & 4: Fetching EPS Data...")
    annuals = stock.financials.T
    quarters = stock.quarterly_financials.T
    
    # Timezone cleanup
    if annuals.index.tz is not None: annuals.index = annuals.index.tz_localize(None)
    if quarters.index.tz is not None: quarters.index = quarters.index.tz_localize(None)
    
    # ✅ COLLECT REPORTING DATES (for JSON export)
    # We combine annual and quarterly dates, sort them, and remove duplicates
    reporting_dates = sorted(list(set(annuals.index) | set(quarters.index)))

    eps_col = None
    for col in ['Diluted EPS', 'Basic EPS']:
        if col in annuals.columns and col in quarters.columns:
            eps_col = col
            break
            
    if not eps_col:
        print(f"⚠️ Could not find EPS column.")
        return None, []

    # --- Step 5: Calculate TTM EPS & Growth ---
    print("\nStep 5: Calculating TTM EPS & Growth Rate...")
    
    # Baseline History
    series_annual_ttm = annuals[eps_col].dropna()

    # Rolling Calculation (4 quarters)
    quarters.sort_index(inplace=True)
    series_quarterly_ttm = quarters[eps_col].rolling(window=4).sum().dropna()
    
    # Combine Data
    s_combined_eps = pd.concat([series_annual_ttm, series_quarterly_ttm])
    s_combined_eps = s_combined_eps[~s_combined_eps.index.duplicated(keep='last')].sort_index()
    
    df_metrics = s_combined_eps.to_frame(name='TTM_EPS')

    # --- Step 7: Merging and Filling ---
    print("Step 7: Merging and Filling...")
    df_metrics_daily = df_metrics.reindex(df_price.index, method='ffill')
    df_final = df_price.join(df_metrics_daily)
    
    # --- Step 8: Calculate PE & PEG ---
    print("Step 8: Calculating PE & PEG Ratios...")
    df_final['PE_Ratio'] = df_final['Close'] / df_final['TTM_EPS']
    
    # Calculate EPS Growth YoY after merge and fill
    df_final['EPS_Growth_YoY'] = (
        (df_final['TTM_EPS'] / df_final['TTM_EPS'].shift(365) - 1) * 100
    )

    # PEG calculation
    df_final['PEG_Ratio'] = df_final['PE_Ratio'] / df_final['EPS_Growth_YoY']
    
    # Cleanup
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.dropna(subset=['PE_Ratio'], inplace=True)
    
    print_debug_view(df_final, "Final Dataframe")
    
    return df_final, reporting_dates

def export_to_json(df, ticker, reporting_dates):
    """
    Exports rows corresponding to the company's specific Quarter/Year End dates.
    Enforces a strict logical order for JSON keys (Metadata -> Price -> EPS -> Valuation -> Financials).
    """
    print(f"\n--- 💾 Exporting JSON for {ticker} ---")
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    # Define the desired logical order for the JSON output
    LOGICAL_ORDER = [
        # Metadata
        "Report_Date_Official", "Trading_Date_Used", "Data_Source_Type",
        
        # Market Data
        "Close",
        
        # EPS & Growth
        "TTM_EPS", "EPS_Growth_YoY",
        
        # Valuation
        "PE_Ratio", "PEG_Ratio",
        
        # Income Statement
        "Revenue", "EBIT", "Net_Income",
        
        # Margins & Returns
        "Net_Margin_Pct", "RONW_Pct",
        
        # Balance Sheet & Debt
        "Total_Debt", "Equity", "Debt_to_Equity", "Interest_Coverage"
    ]
    
    export_data = []
    
    # Filter dates to those within our available price history range
    valid_dates = [d for d in reporting_dates if d >= df.index.min() and d <= df.index.max()]
    
    for report_date in valid_dates:
        try:
            # Find closest trading day (asof/pad logic)
            loc = df.index.get_indexer([report_date], method='pad')[0]
            
            if loc != -1:
                # 1. Get raw data from DataFrame
                row_data = df.iloc[loc].to_dict()
                
                # 2. Add/Inject Metadata
                row_data['Report_Date_Official'] = report_date.strftime('%Y-%m-%d')
                row_data['Trading_Date_Used'] = df.index[loc].strftime('%Y-%m-%d')
                
                # Ensure Data_Source_Type exists (defaulting if not in df)
                if 'Data_Source_Type' not in row_data:
                    row_data['Data_Source_Type'] = "Quarterly_TTM"

                # 3. Clean Data (Handle Timestamps & NaN)
                clean_row = {}
                for k, v in row_data.items():
                    if isinstance(v, (pd.Timestamp, datetime)):
                        clean_row[k] = v.strftime('%Y-%m-%d')
                    elif pd.isna(v):
                        clean_row[k] = None
                    else:
                        clean_row[k] = v
                
                # 4. Reorder Dictionary based on LOGICAL_ORDER
                ordered_row = {}
                
                # First, add keys specifically defined in our logical order
                for key in LOGICAL_ORDER:
                    if key in clean_row:
                        ordered_row[key] = clean_row[key]
                
                # Then, append any remaining keys (dynamic columns) that weren't in the list
                for key in clean_row:
                    if key not in ordered_row:
                        ordered_row[key] = clean_row[key]

                export_data.append(ordered_row)

        except Exception as e:
            print(f"Warning skipping date {report_date}: {e}")

    filename = f"fundamentals_{ticker}.json"
    filepath = os.path.join(OUTPUTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=4)
        
    print(f"✅ Data exported to: {filepath}")

def create_charts(ticker):
    df, reporting_dates = get_fundamental_data(ticker)
    if df is None or len(df) < 50: return None, None

    # --- Export Data Hook ---
    export_to_json(df, ticker, reporting_dates)

    print("Step 9: Plotting...")
    
    mean_pe = df['PE_Ratio'].mean()
    std_pe = df['PE_Ratio'].std()
    
    # --- CHART 1: Quantamental (PE & EPS) ---
    fig_fund = make_subplots(specs=[[{"secondary_y": True}]])
    fig_fund.add_trace(go.Scatter(x=df.index, y=df['TTM_EPS'], name="TTM EPS", fill='tozeroy', line=dict(width=0), fillcolor='rgba(46, 204, 113, 0.2)'), secondary_y=False)
    fig_fund.add_trace(go.Scatter(x=df.index, y=df['PE_Ratio'], name="PE Ratio", line=dict(color='#2980b9', width=2)), secondary_y=True)
    
    fig_fund.add_hline(y=mean_pe, line_dash="dot", line_color="black", secondary_y=True, annotation_text="Mean PE")
    fig_fund.add_hline(y=mean_pe + std_pe, line_dash="dash", line_color="green", secondary_y=True)
    fig_fund.add_hline(y=mean_pe - std_pe, line_dash="dash", line_color="green", secondary_y=True)
    fig_fund.add_hline(y=mean_pe + 2*std_pe, line_dash="dash", line_color="red", secondary_y=True)
    fig_fund.add_hline(y=mean_pe - 2*std_pe, line_dash="dash", line_color="red", secondary_y=True)

    # Set the range for the primary y-axis (TTM_EPS)
    fig_fund.update_yaxes(
        range=[df['TTM_EPS'].min() * 0.9, df['TTM_EPS'].max() * 1.1],  # Adjust multiplier as needed
        secondary_y=False
    )

    fig_fund.update_layout(title=f"<b>{ticker} Quantamental Chart (PE & EPS)</b>", template="plotly_white", height=500)

    # --- CHART 2: Technicals + PEG Ratio ---
    df['RSI'] = calculate_rsi(df['Close'])
    
    # We define specs to allow a secondary axis on the first row (Price Chart)
    fig_tech = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]] # Row 1 gets 2 axes, Row 2 gets 1
    )
    
    # Row 1, Axis 1: Price
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='black')), row=1, col=1, secondary_y=False)
    
    # Row 1, Axis 2: PEG Ratio (The new requirement)
    # We use a dotted orange line for PEG to distinguish it clearly
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['PEG_Ratio'], name="PEG Ratio", line=dict(color='#e67e22', width=1, dash='dot')), row=1, col=1, secondary_y=True)
    
    # Row 2, Axis 1: RSI
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
    fig_tech.add_hrect(y0=30, y1=70, row=2, col=1, fillcolor="blue", opacity=0.1, line_width=0)
    
    # Update Layout for dual axis
    fig_tech.update_layout(
        title=f"<b>{ticker} Technicals + PEG Ratio</b>", 
        template="plotly_white", 
        height=600, 
        showlegend=True,
        yaxis2=dict(title="PEG", overlaying='y', side='right', showgrid=False) # Customize secondary y-axis label
    )

    return fig_fund, fig_tech

def save_charts_to_html(fig1, fig2, filename="charts_TEST.html"):
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
        
    filepath = os.path.join(OUTPUTS_DIR, filename)
    with open(filepath, 'w') as f:
        f.write("<html><head><title>Quantamental Charts</title></head><body>")
        
        # ✅ ORDER SWAPPED: Fig 2 (Technicals) first, then Fig 1 (Fundamentals)
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<br><hr><br>")
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        
        f.write("</body></html>")
    print(f"\n✅ Charts saved to: {filepath}")

if __name__ == "__main__":
    ticker = input("Enter ticker (e.g., RELIANCE.NS or AAPL): ").strip()
    f1, f2 = create_charts(ticker)
    if f1 and f2:
        save_charts_to_html(f1, f2, f"charts_{ticker}.html")
        f1.show()
        f2.show()