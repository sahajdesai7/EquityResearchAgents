import sys
import os

# --- Import your modules ---
# We point to the 'modules' folder and use 'as' to create an alias.
# This allows the rest of your code to remain exactly the same.
import modules.screener as screener
import modules.charts as charts
import modules.analyst as analyst
import modules.forensic as forensic
import modules.reporter as reporter

def main():
    print("🚀 STARTING AI EQUITY RESEARCH PIPELINE")
    print("=======================================")

    # --- STEP 1: SCREENER ---
    # This function asks the user for Region, scans stocks, 
    # and returns the specific ticker the user wants to analyze.
    ticker = screener.screen_stocks()

    if not ticker:
        print("\n❌ No ticker selected or no results found. Exiting pipeline.")
        sys.exit()

    print(f"\n✅ Target Acquired: {ticker}")
    print("   Proceeding to Deep Dive Analysis...")

    # --- STEP 2: CHARTS ---
    # We generate the charts first so the file exists for the Reporter later.
    print("\n📈 STEP 2: Generating Technical & Quantamental Charts...")
    try:
        fig1, fig2 = charts.create_charts(ticker)
        if fig1 and fig2:
            # We use a consistent naming convention so Reporter can find it
            chart_filename = f"charts_{ticker}.html"
            charts.save_charts_to_html(fig1, fig2, chart_filename)
        else:
            print("   ⚠️ Not enough data to generate charts.")
    except Exception as e:
        print(f"   ❌ Error generating charts: {e}")

    # --- STEP 3: ANALYST AGENT (Qualitative) ---
    print("\n🕵️ STEP 3: Running Market Analyst Agent (News & Sentiment)...")
    try:
        analyst.analyze_sentiment_and_news(ticker)
    except Exception as e:
        print(f"   ❌ Error in Analyst module: {e}")

    # --- STEP 4: FORENSIC AGENT (Quantitative) ---
    print("\n🧪 STEP 4: Running Forensic Accountant Agent (Earnings Quality)...")
    try:
        forensic.analyze_earnings_quality(ticker)
    except Exception as e:
        print(f"   ❌ Error in Forensic module: {e}")

    # --- STEP 5: REPORTER AGENT (Final Output) ---
    print("\n🏆 STEP 5: Generating Final Investment Memo...")
    try:
        # This module looks for the files created in Steps 2, 3, and 4
        final_report_path = reporter.generate_investment_memo(ticker)
        
        print("\n" + "="*50)
        print(f"🎉 PIPELINE COMPLETE!")
        print(f"📄 Final Investment Memo: {final_report_path}")
        print("="*50)
        
        # Optional: Automatically open the report (Works on Windows/macOS)
        if final_report_path and os.path.exists(final_report_path):
            try:
                if os.name == 'nt': # Windows
                    os.startfile(final_report_path)
                elif os.name == 'posix': # macOS/Linux
                    os.system(f"open '{final_report_path}'")
            except:
                pass

    except Exception as e:
        print(f"   ❌ Error in Reporter module: {e}")

if __name__ == "__main__":
    main()