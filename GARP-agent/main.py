import sys
import os
import time

# --- Import your modules ---
# We assume these files are in a 'modules' subdirectory.
# We use 'as' to create a clean alias for calling functions.
import modules.screener as screener
import modules.charts as charts
import modules.analyst as analyst
import modules.forensic as forensic
import modules.reporter as reporter

def main():
    print("🚀 STARTING AI EQUITY RESEARCH PIPELINE")
    print("=======================================")

    # --- STEP 1: SCREENER ---
    # User selects a ticker from the generated list
    ticker = screener.screen_stocks()

    if not ticker:
        print("\n❌ No ticker selected or no results found. Exiting pipeline.")
        sys.exit()

    print(f"\n✅ Target Acquired: {ticker}")
    print("   Proceeding to Deep Dive Analysis...")

    # --- STEP 2: CHARTS & VALUATION DATA ---
    # CRITICAL: This step generates 'fundamentals_{ticker}.json' which is REQUIRED
    # by the Reporter Agent in Step 5 for the scoring model.
    print(f"\n📈 STEP 2: Generating Technicals & Valuation Data...")
    try:
        fig1, fig2 = charts.create_charts(ticker)
        if fig1 and fig2:
            chart_filename = f"charts_{ticker}.html"
            charts.save_charts_to_html(fig1, fig2, chart_filename)
            print("   ✅ Charts & JSON Valuation Data created.")
        else:
            print("   ⚠️ Not enough data to generate charts/JSON.")
    except Exception as e:
        print(f"   ❌ Error generating charts: {e}")

    # --- STEP 3: ANALYST AGENT (Qualitative) ---
    # Generates 'research_{ticker}.txt'
    print(f"\n🕵️ STEP 3: Running Market Analyst Agent...")
    try:
        analyst.analyze_sentiment_and_news(ticker)
    except Exception as e:
        print(f"   ❌ Error in Analyst module: {e}")

    # --- STEP 4: FORENSIC AGENT (Quantitative) ---
    # Generates 'forensic_{ticker}.txt'
    print(f"\n🧪 STEP 4: Running Forensic Accountant Agent...")
    try:
        forensic.analyze_earnings_quality(ticker)
    except Exception as e:
        print(f"   ❌ Error in Forensic module: {e}")

    # --- STEP 5: REPORTER AGENT (Final Output) ---
    # Consumes: fundamentals.json, research.txt, forensic.txt, charts.html
    # Produces: Investment_Memo_{ticker}.html
    print(f"\n🏆 STEP 5: Generating Final Investment Memo...")
    try:
        final_report_path = reporter.generate_investment_memo(ticker)
        
        print("\n" + "="*50)
        print(f"🎉 PIPELINE COMPLETE!")
        print(f"📄 Final Investment Memo: {final_report_path}")
        print("="*50)
        
        # Optional: Automatically open the report
        if final_report_path and os.path.exists(final_report_path):
            try:
                if os.name == 'nt': # Windows
                    os.startfile(final_report_path)
                elif os.name == 'posix': # macOS/Linux
                    os.system(f"open '{final_report_path}'")
            except Exception:
                pass

    except Exception as e:
        print(f"   ❌ Error in Reporter module: {e}")

if __name__ == "__main__":
    main()